import os
import sys
import time
from collections import OrderedDict
from time import strftime, gmtime
from tensorboardX import SummaryWriter
from dataset import AsrDataset, DataLoader, AsrCollator
from models.transformer import Model
import hparams as hp
import argparse
import megengine as mge
import megengine.module as M
import megengine.functional as F
from megengine.functional import clip, concat, minimum, norm
from megengine.core._imperative_rt.core2 import pop_scope, push_scope
from typing import Iterable, Union
from megengine.tensor import Tensor
import megengine.distributed as dist
from megengine.data import SequentialSampler, RandomSampler, DataLoader
from criterions.label_smoothing_loss import LabelSmoothingLoss
from megengine.utils.network import Network as Net
import megengine.autodiff as autodiff
import megengine.data as data
import megengine
import multiprocessing

logging = megengine.logger.get_logger()


def clip_grad_norm(
    tensors: Union[Tensor, Iterable[Tensor]],
    max_norm: float,
    ord: float = 2.0,
):
    push_scope("clip_grad_norm")
    if isinstance(tensors, Tensor):
        tensors = [tensors]
    tensors = [t for t in tensors if t.grad is not None]
    norm_ = [norm(t.grad.flatten(), ord=ord) for t in tensors]
    if len(norm_) > 1:
        norm_ = norm(concat(norm_), ord=ord)
    else:
        norm_ = norm_[0]
    scale = max_norm / (norm_ + 1e-6)
    scale = minimum(scale, 1)
    for tensor in tensors:
        tensor.grad._reset(tensor.grad * scale)
    pop_scope("clip_grad_norm")
    return norm_


class exponential_ma:
    def __init__(self, ratio):
        self.value = 0
        self.weight = 0
        self.ratio = ratio

    def update(self, x):
        self.value = self.value * self.ratio + (1 - self.ratio) * x
        self.weight = self.weight * self.ratio + (1 - self.ratio)

    def get_value(self):
        if self.weight < 1e-8:
            return 0
        return self.value / self.weight


def update_train_log(monitor_vars_name, ma_dict, losses, ttrain, tdata):
    for n in monitor_vars_name:
        for ma in ma_dict["losses"]:
            ma[n].update(losses[n])
    for ma in ma_dict["ttrain"]:
        ma.update(ttrain)
    for ma in ma_dict["tdata"]:
        ma.update(tdata)


def print_train_log(sess, epoch, minibatch, ma_dict, minibatch_per_epoch):
    ma_output = "[{}] e:{}, {}/{} ".format(
        strftime("%Y-%m-%d %H:%M:%S", gmtime()), epoch, minibatch, minibatch_per_epoch
    )
    print(ma_output, file=sys.stderr)
    line = "    {:31}:".format("speed")
    for ma in ma_dict["ttrain"]:
        line += "{:10.2g}".format(1 / ma.get_value())
    print(line, file=sys.stderr)
    line = "    {:31}".format("dp/tot")
    for ma1, ma2 in zip(ma_dict["ttrain"], ma_dict["tdata"]):
        line += "{:10.2g}".format(ma2.get_value() / ma1.get_value())
    print(line, file=sys.stderr)
    for k in sess.loss_names:
        line = "    {:31}".format(k)
        for ma in ma_dict["losses"]:
            line += "{:10.2E}".format(ma[k].get_value())
        print(line, file=sys.stderr)
    line = "    {:31}:    {}".format("lr", sess.get_learning_rate())
    print(line, file=sys.stderr)
    sys.stderr.flush()


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = (
        hp.lr
        * warmup_step ** 0.5
        * min(step_num * warmup_step ** -1.5, step_num ** -0.5)
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def set_grad(net, min, max):
    for param in net.parameters():
        param.grad = mge.random.uniform(min, max, param.shape)
        param.grad_backup = F.copy(param.grad)


class Session:
    def __init__(self, args):
        with open(os.path.join(hp.dataset_root, "vocab.txt")) as f:
            self.vocab = [w.strip() for w in f.readlines()]
            self.vocab = ["<pad>"] + self.vocab
            print(f"Vocab Size: {len(self.vocab)}")
        self.model = Model(hp.num_mels, len(self.vocab))
        world_size = args.world_size * args.ngpus
        if world_size > 1:
            dist.bcast_list_(self.model.parameters(), dist.WORLD)

        # Autodiff gradient manager
        self.gm = autodiff.GradManager().attach(
            self.model.parameters(),
            callbacks=dist.make_allreduce_cb("SUM") if world_size > 1 else None,
        )
        self.global_step = 0
        self.optimizer = mge.optimizer.Adam(self.model.parameters(), lr=hp.lr)

        # load pretrain model
        if args.continue_path:
            ckpt = mge.load(args.continue_path)
            if "model" in ckpt:
                state_dict = ckpt["model"]
                self.model.load_state_dict(state_dict, strict=False)

        self.loss_names = ["total"]
        self.criterion = LabelSmoothingLoss(len(self.vocab), 0, hp.lsm_weight)

    def get_learning_rate(self):
        lr = self.optimizer.param_groups[0]["lr"]
        return lr

    def get_current_losses(self):
        losses = OrderedDict()
        for name in self.loss_names:
            losses[name] = float(getattr(self, "loss_" + name))
        return losses

    def optimize_parameters(self, data):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        text_input, text_output, mel, pos_text, pos_mel, text_length, mel_length = data
        with self.gm:
            hs_pad, hs_mask, pred_pad, pred_mask = self.model.forward(
                mel, mel_length, text_input, text_length
            )
            self.loss_total = self.criterion(pred_pad, text_output)
            self.gm.backward(self.loss_total)
        clip_grad_norm(self.model.parameters(), 1.0)
        self.optimizer.step().clear_grad()


def main():
    os.makedirs(hp.checkpoint_path, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--continue_path")

    parser.add_argument(
        "-n",
        "--ngpus",
        default=None,
        type=int,
        help="number of GPUs per node (default: None, use all available GPUs)",
    )
    parser.add_argument(
        "--save",
        metavar="DIR",
        default="output",
        help="path to save checkpoint and log",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        help="number of total epochs to run (default: 90)",
    )

    parser.add_argument("-j", "--workers", default=2, type=int)
    parser.add_argument(
        "-p",
        "--print-freq",
        default=20,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument("--dist-addr", default="localhost")
    parser.add_argument("--dist-port", default=23456, type=int)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)

    args = parser.parse_args()

    # create server if is master
    if args.rank <= 0:
        server = dist.Server(
            port=args.dist_port
        )  # pylint: disable=unused-variable  # noqa: F841

    # get device count
    with multiprocessing.Pool(1) as pool:
        ngpus_per_node, _ = pool.map(megengine.get_device_count, ["gpu", "cpu"])
    if args.ngpus:
        ngpus_per_node = args.ngpus
    # launch processes
    procs = []
    for local_rank in range(ngpus_per_node):
        p = multiprocessing.Process(
            target=worker,
            kwargs=dict(
                rank=args.rank * ngpus_per_node + local_rank,
                world_size=args.world_size * ngpus_per_node,
                ngpus_per_node=ngpus_per_node,
                args=args,
            ),
        )
        p.start()
        procs.append(p)

    # join processes
    for p in procs:
        p.join()


def worker(rank, world_size, ngpus_per_node, args):

    # pylint: disable=too-many-statements
    if rank == 0:
        os.makedirs(os.path.join(args.save, "asr"), exist_ok=True)
        megengine.logger.set_log_file(os.path.join(args.save, "asr", "log.txt"))
    # init process group
    if world_size > 1:
        dist.init_process_group(
            master_ip=args.dist_addr,
            port=args.dist_port,
            world_size=world_size,
            rank=rank,
            device=rank % ngpus_per_node,
            backend="nccl",
        )
        logging.info(
            "init process group rank %d / %d", dist.get_rank(), dist.get_world_size()
        )

    # build dataset
    train_dataloader = build_dataset(args)
    train_queue = iter(train_dataloader)
    steps_per_epoch = 164905 // (world_size * hp.batch_size)
    sess = Session(args)
    ma_rates = [1 - 0.01 ** x for x in range(3)]
    ma_dict = {
        "losses": [
            {k: exponential_ma(rate) for k in sess.loss_names} for rate in ma_rates
        ],
        "ttrain": [exponential_ma(rate) for rate in ma_rates],
        "tdata": [exponential_ma(rate) for rate in ma_rates],
    }
    for epoch in range(1, (hp.epochs + 1) * steps_per_epoch):
        t_minibatch_start = time.time()
        sess.global_step += 1
        if sess.global_step < 400000:
            adjust_learning_rate(sess.optimizer, sess.global_step)
        tdata = time.time() - t_minibatch_start
        data = next(train_queue)
        sess.optimize_parameters(data)
        losses = sess.get_current_losses()
        ttrain = time.time() - t_minibatch_start
        # print(ttrain, tdata)
        update_train_log(sess.loss_names, ma_dict, losses, ttrain, tdata)
        if sess.global_step % hp.log_interval == 0 and rank == 0:
            print_train_log(sess, epoch, epoch, ma_dict, hp.epochs * steps_per_epoch)
        if sess.global_step % hp.save_interval == 0 and rank == 0:
            print("*******************************************")
            mge.save(
                {"model": sess.model.state_dict(), "global_step": sess.global_step},
                os.path.join(
                    hp.checkpoint_path, "checkpoint_%d.pkl" % sess.global_step
                ),
            )
            print("*******************************************")
        if sess.global_step > hp.max_steps:
            exit(1)


def build_dataset(args):

    dataset = AsrDataset()
    train_sampler = data.Infinite(
        RandomSampler(dataset=dataset, batch_size=hp.batch_size)
    )
    dataloader = DataLoader(
        dataset=dataset, sampler=train_sampler, collator=AsrCollator()
    )

    return dataloader


if __name__ == "__main__":
    main()
