### 1. 简化结构

- 数据预处理：将图像转为灰度图像，再归一化（或者将图像归一化后再转为灰度图像）
- 模型：整理为一个TextRecognition对象，使用TPS-ResNet-BiLSTM-Attn-case-sensitive.pth模型（下载地址：[TRBA (case-sensitive version)](https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY)）。**需要注意的是，该项目将model = nn.DataParallel(model)保存为模型，为了方便，我们需要重新将model.module保存为模型**

测试用例，对demo_1.png,demo_3.png,demo_5.png,demo_7.png分别归一化
再cat为[4,3,32,100]的张量，然后将之转为灰度图像的张量，维度为[4,1,32,100]，再进行文字识别：

```bashrc
$ python text_recognition.py \
--model_path 'pretrained/model.pth'
```

输出结果为：

    pred:  Available, confidence_score:  0.9996113181114197
    pred:  Londen, confidence_score:  0.6106716394424438
    pred:  TOAST, confidence_score:  0.9872652292251587
    pred:  underground, confidence_score:  0.9998375177383423

### 2. 与StyleText推理部分结合

对i_s和o_f进行文字识别，其中i_s经过预处理，o_f为styletext模型预测结果：

```bashrc
$ python predictors.py \
--model_path 'pretrained/model.pth'
```
输出结果为：

    pred:  Available, confidence_score:  0.9917135834693909
    pred:  YANG, confidence_score:  0.9953127503395081

### 3. 依赖

- 运行环境：PyTorch 1.9, CUDA 10.2, python 3.8, CentOS 7

- requirements : pillow, torchvision, pygame==2.0.0, opencv-python, numpy