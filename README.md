# -Unet-

项目主要完成的任务是对天文图像的降噪处理

采用的方法是先在图片上产生噪声，将噪声图片视为训练集，原图像视为标签，通过Unet网络进行训练

训练平台是google colab

### 产生数据

数据集在npy文件夹下。产生原始数据的函数为generate_data.py

通过generate_data.py在npy数据集中寻找所有黑色率小于0.35的204张图片。图片的列表在figurelist.txt中

同时产生一个(204,400,400)的npy文件origin.npy

### 训练

先在本地文件Unet.py中写好训练的代码。

并通过tensorboard产生logs文件，在tensorboard提供的本地网页得到可视化的网络结构。

之后再google colab中上传origin.npy数据集，并在google colab用本笔记中的代码进行训练。

得到traing_Time43.h5网络模板文件

### 预测

再次使用generate_data.py文件避开之前采用的npy图片产生evallist.txt文件得到一些可用于预测的文件。

通过predict.py 读取evallist.txt得到数据，用和训练类似的方法产生噪声文件，并将预测的结果和噪声图片放入result文件夹中

通过result文件进行数据可视化

以及训练数据GolbalLoss.txt文件

### 数据可视化

先画出了训练中的Loss函数

可以任意读取result中predict的数据

通过Show函数得到图像在1-50部分的像素分布

通过pltshow展示图片，接下来直观地展示产生的噪声和去噪之后的结果。

### 先进性分析

Unet本身多用于处理图像的分割和语义分析，本项目则尝试了使用Unet进行天文图像的去噪。

不同于一般的Unet，在Unet的网络结构中加入了含有曝光时间的张量，增加了项目的鲁棒性。

最后一层也没有采用传统的‘sigmod’函数而是直接传递全连接的结果，把原图作为训练数据的标签集。

采用tensorflow2.0中的layers层， 可以较为完整地看到网络的内在搭建逻辑。并使用了tensorboard对网络结构进行了可视化。