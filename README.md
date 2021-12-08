# END TO END ASR TRANSFORMER
* 本项目基于transformer 6\*encoder+6\*decoder的基本结构构造的端到端的语音识别系统

## Model

<img src="egs/model.pbm" width=50%/>

## Instructions
* 1.数据准备:
    * 自行下载数据，遵循文件结构如下：
```
├── data
│   ├── train
│   ├── dev
│   ├── test
``` 
* 2.数据预处理：
    * 运行`prepare_data.py`对数据进行预处理, 获得整个词表，每个样本音频的`mel-scale-spectrogram`，文本的`token-ids`
* 3.模型训练：
    * 运行`train_transformer.py --ngpus 8`进行transformer网络的训练. 该网络输入`mel-scale-spectrogram`, 输出`token-ids`
* 4.模型推理：
    * 运行`evlauate.py`在dev/test上测试准确率


## Acknowledgements
* [GitHub ESPET](https://github.com/espnet/espnet/tree/master/espnet) 
* [GitHub Speech-Transformer-tf2.0](https://github.com/xingchensong/Speech-Transformer-tf2.0)
* [Github speech-transformer](https://github.com/sooftware/speech-transformer)

## Reference
* Ashish Vaswani et al. “[Attention Is All You Need](https://arxiv.org/abs/1706.03762)”  (2017). 
* Abdel-rahman Mohamed et al. “[Transformers with convolutional context for ASR](https://arxiv.org/abs/1904.11660)” arXiv: Computation and Language (2019).
* Albert Zeyer et al. “Improved Training of End-to-end Attention Models for Speech Recognition” Conference of the International Speech Communication Association (2018).

