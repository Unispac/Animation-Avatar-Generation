# 基于GAN的动漫头像生成



## 运行

实验环境：<br>		tensorflow-gpu-1.18.0<br>		cuda 10.0.130<br>		cudnn  7.5.0.56



代码在code目录下，pretrain-model较大，上传到了浙大云盘上，可以在[链接处](https://pan.zju.edu.cn/share/a2e12eaf246b43106a94324d6d)下载，校内访问应该可以达到10mb/s的传输速度。

[数据集](https://pan.zju.edu.cn/share/fb26846803a2db2e89c503c48b)也一并上传到了浙大云盘上。

How to run ? 

训练： python main.py

生成：python main.py --mode infer
