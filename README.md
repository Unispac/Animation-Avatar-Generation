[TOC]

# 基于GAN的动漫头像生成



## 运行

实验环境：<br>		tensorflow-gpu-1.18.0<br>		cuda 10.0.130<br>		cudnn  7.5.0.56



代码在code目录下，pretrain-model较大，上传到了浙大云盘上，可以在[链接处](https://pan.zju.edu.cn/share/a2e12eaf246b43106a94324d6d)下载，校内访问应该可以达到10mb/s的传输速度。

[数据集](https://pan.zju.edu.cn/share/fb26846803a2db2e89c503c48b)也一并上传到了浙大云盘上。

How to run ? 

训练： python main.py

生成：python main.py --mode infer



## 项目来源

本项目实践了基于GAN的动漫头像生成。

想法来源于一个名为 [MakeGirlsMoe](https://make.girls.moe/#/) 的动漫人物生成项目：

![MakeGirlsMoe](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/1.png)

该项目是由一组来自复旦大学和CMU的学生设计的，曾经一度在github trend上跻身Top 5，并且登上过日本雅虎的首页。项目基于GAN的图像生成技术，允许用户设置特征参数控制动漫人物的自动生成。

并且，项目原作者在最新发布的项目[Crypko](https://crypko.ai/#/)中更是实现了人物连续变化的精确控制，使得通过GAN生成风格连贯的动画成为可能：

![生成](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/2.png)

除此之外，这一角色生成技术在2D绘图辅助设计上也表现不俗：

![辅助设计](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/3.png)

可以预见，在动画影视产业高速增长的今天，一个成熟的自动作画/辅助作画的AI系统有着非常大的市场空间。



## 项目内容概述

在实现这个项目的过程中，我们通过阅读原论文和一些辅助材料，对GAN的技术有了一个初步的了解。受限于时间和人力，我们只实现了64*64动漫头像生成，先后基于DCGAN和ACGAN的方法实现了无条件生成和条件约束生成。

数据集来自台湾大学李宏毅老师开设的课程：MLDS。

在实验中，我们验证和研究了GAN在理论上存在的一些问题以及潜在的解决方法。我们使用原论文中提供的算法生成图像时也遇到了很多问题，通过一些分析，我们提出了一些改进手段，最后在一定程度优化了生成结果，实现了可接受的生成效果：

![DEMO](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/4.png)

这篇实验报告中，我们将会简单的介绍我们的实验内容和遇到的问题，给出一些分析和解决方法。最终我们将分析实验结果及其潜在的优化空间。



## GAN及其固有的问题

生成式对抗网络是2014年由Goodfellow等人在《Generative Adversarial Net》中提出的。

在GAN中一组Generator和Discriminator通过对下面这个式子进行minimax博弈，实现生成准确度的自强化学习：

![minimax Loss](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/5.png)

原论文对此做了一个简单的证明，当Discriminator达到自己的纳什均衡的时候，Generator的目标函数等价于生成器产生结果的分布$p_g$和训练集数据分布$p_{data}$的JS散度。于是从理论上，我们可以得到G和D的对抗博弈本质上式一个$p_g$向$p_{data}$拟合的过程。

相对于经典的VAE方法，GAN摒弃向量差范数这种相对机械的评估方法，而采用了一个Discriminator网络来做出更加复杂的评估，因此Generator也能更好的学到像素点之间的correlation。

Goodfellow提出的初代GAN版本在应用中取得了很多出色的表现，但是也一直存在难以训练的问题。在训练的时候经常会发现当Discriminator训练得太好的时候，Generator训练时就会出现梯度消失的现象。Goodfellow也原文中也提到过这个问题，错误的把原因归结为saturation，并且提出把generator训练时使用的log(1-D(G(z)))换成-log(D(G(z)))，结果是这个替代选项经常出现梯度不稳定的现象。

从2014年开始，就不断有各种各样的关于GAN的研究，大多数声称对于原始版本GAN的改进都是通过实验挑选出更加好的网络架构和超参数设置，但是都没有彻底解决上述问题。

2017年Arjovsky发表的两篇论文《TOWARDS PRINCIPLED METHODS FOR TRAINING GENERATIVE ADVERSARIAL NETWORKS》和 《Wasserstein GAN》引起了广泛的关注。前者从理论上给出了上述问题出现的原因，后者提出了WGAN来解决这个问题。

Arjovsky从数学上证明了JS散度对于$p_{data}$和$p_g$之间的距离衡量在两者重叠部分测度为0的时候是一个常数。而由于$p_{data}$和$p_g$的支撑集在高维图像空间中都是低维流形，因此两者重叠部分的期望测度也的确是0。。。这正是用JS散度来作为G的博弈均衡目标时梯度消失的根本原因。同时Goodfellow给出的替代目标函数在经过简单的数学变换后呈现处KL-JS的形式，其实是一对互相矛盾的目标，因此才会造成梯度不稳定。

因此，Arjovsky提出用Wasserstein Distance来替代JS散度。虽然Wasserstein Distance不可直接结算，Arjovsky使用了一个数学定理证明了其等价的对偶式，最后提出将Discriminator替换成一个拟合这个对偶式的网络，并通过weight clip的手段来满足这个对偶式必需的Lipschitz条件。

尽管后来有诸如WGAN-GP这样的方法提出用Gradient Penalty更好的来满足Lipschitz条件，但是我们在实验中发现，WGAN-GP虽然可能克服了DCGAN容易出现梯度消失的问题，但是在DCGAN正常训练的情况下，WGAN-GP和WGAN的生成的效果却不如DCGAN。

![DCGAN VS WGAN-GP](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/6.png)

在采用同样的架构（WGAN-GP中仅仅是把Discriminator中的norm层去掉，且输出不再加上sigmoid限制）时，我们发现WGAN-GP生成的结果中，颜色强度很大，生成的结果不够柔和，清晰度也不如成功训练出来的DCGAN。我们认为可能存在的问题主要有：WGAN-GP中wasserstein distance对偶式的拟合需要更加精心设计的网络；需要更好的调整参数保持一个适当的Lipschitz常数。由于时间原因，我们最终没办法对这些改进的想法做验证。

在对GAN进行学习和研究的时候，我们也整理了另外两篇文档来详细的讨论GAN的思想和理论推导，分别收录在：[GAN的基本想法](http://122.152.198.128/?p=1540)，[GAN的理论](http://122.152.198.128/?p=1550)。



## 无条件生成

无条件生成时，我们采用了一个DCGAN架构实验。

采用的基本的架构如下：

![DCGAN架构](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/7.png)



**实验时采用的几个trick：**

- Discriminator 和 Generator 都加上 BatchNormalization。

- 训练图片的通道分量统一除以127.5后减1，限制在-1和1之间，产生的图片由tanh将分量约束在-1~1之间。

- 输入的图片会以一定概率分布被随机的左右翻转以及正逆时针轻微旋转，培养模型对对称性的理解。

- Z的从一个均值为0，标准差为$e^{\frac{-1}{\Pi}}$的正态分布中取样。

	

**生成效果：**

- After 1 Epoch

![1 Epoch](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/8.png)

- After 3 Epoches

![3 Epoches](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/9.png)

- After 10 Epoches

![10 Epoches](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/10.png)

- After 30 Epoches

![30 Epoches](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/11.png)

- After 50 Epoches

![50 Epoches](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/12.png)



## 条件生成

在无条件生成中，GAN做的是拟合由z分布以及generator参数共同隐式定义的生成分布$g_p$和数据集分布$d_{data}$。因此，生成器能做到的只是给定一个z分布中采样的向量，然后生成一个尽可能真实的图片。

Conditional GAN允许我们提供条件，对条件概率拟合，生成符合条件的结果。我们使用ACGAN实现了生成过程中对头发颜色以及眼睛颜色的控制。

### ACGAN

我们采用了ACGAN（《Conditional Image Synthesis With Auxiliary Classifier GANs》, Odena A et. al 2017）来实现。

![ACGAN基本模式](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/13.png)

![Loss Function for ACGAN](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/14.png)

如上所示，ACGAN的想法非常简单，Discriminator除了被用来给真实度评分，其提取的图像特征还被用来做分类，判定输入图片的类别。生成器除了提供一个噪音，还需要提供一个类条件。

Loss Function在无条件GAN的基础上添加了一个分类损失项，训练Discriminator的时候，会引导Discriminator对数据集中图片做出正确分类，训练Generator的时候会引导生成器产生与类条件一致的结果。

值得一提的是，原论文中没有考虑lambda的取值，我们添加了一个lambda值来平衡网络对真实度/类别的取舍。我们在实验中发现了lambda会明显的对生成结果产生影响。lambda太大，会导致分类失败，lambda太小，会影响图片生成的质量。

### 架构的选择

我们在原先的DCGAN的discriminator架构上做了一点微调，使得它能产生两种输出：

![第一版ACGAN](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/15.png)

实验结果显示，它成功的做出了分类，但是生成图片的质量却下降了：

![After Enough Epoches](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/16.png)

如图，可以看到生成结果的清晰度严重降低，脸部崩坏的概率也提高了，并且再继续训练下去也没有明显的改变。

考虑到Discriminator和Generator都要处理class的信息，需要拟合的关系变得更加复杂，很有可能是因为原来的网络表示力不够。于是我们参考了采用了 [MakeGirlsMoe](https://make.girls.moe/#/) 项目原作者使用的一套残差网络设计：

![Generator](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/17.png)

![Discriminator](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/18.png)

这套网络其实也是基本上基于SRResNet（《Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network》, Christian Ledig et. al 2017）改造而成的。

但是发现产生的结果分类失败了：

![分类失败样例](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/19.png)

如图，原本每一行应该是一样的发色和眼睛颜色，但是生成结果却没有体现出这种分类效果。不过我们发现图片质量的确非常高，如果仅用随机生成的标准来评判，它应该要优于原来的DCGAN网络结构。

我们推测很有可能是在这种网络设计下，lambda取太大，真实度的loss压过了分类的loss，使得真实度得到了很好的满足，分类却没有很好的得到保障。

我们把网络的lambda取回1后，又不幸的发现，分类的效果虽然恢复了，生成的质量又下降了。

![lambda=1时，质量下降](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/20.png)

把lambda在1~5之间都尝试后，我们发现两者有明显的冲突。质量高了，分类就变弱。分类变好了，质量又弱了。

这让我们意识到ACGAN的设计可能并不合理，ACGAN在判定真实度与判定类别的时共享一个特征提取器。但是这两个任务之间并没有必然的重叠性，就像要判断一个人是男性还是女性和判别一个对象到底是不是人可能用到的特征根本就没有太大交叠性。如果共享网络，则会导致两个任务竞争资源。

于是我们对网络做了一些折中的调整，将ResNet提取的特征分别送入两个更深（3层）的全连接层网络，再取lambda=5时。这样可以降低两个任务对模块依赖的耦合性。让两个任务的区别尽量体现在自己独占的全连接层网络上，让两者的重叠部分又ResNet去解决。

最后，我们得到了较为出色的效果：

![改进后的效果](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/22.png)

### 实验结果展示

250个epoch之后，我们选择了几组特征条件，随机产生：

- Aqua hair and Orange eyes

![Aqua hair and Orange eyes](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/23.png)

- Black hair and Yellow eyes

![Black hair and Yellow eyes](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/24.png)

- Blond hair and Purple eyes

![Blond hair and Purple eyes](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/25.png)

- Blue hair and Blue eyes

![Blue hair and Blue eyes](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/26.png)

- White hair and Gray eyes

![White hair and Gray eyes](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/27.png)

- Pink hair and Orange eyes

![Pink hair and Orange eyes](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/28.png)



## 潜在的优化空间

- 我们发现ACGAN的设计可能并不合理，共享分类和真实度评估可能本来就不是一个好的主义。如果不考虑时间和空间，用两个网络分别分类和评估真实度应该可以期望得到更好的效果。采用其它设计类型的CGAN模型也是一个很好的选择。

- 采用更好的数据集。我们发现 [MakeGirlsMoe](https://make.girls.moe/#/) 这个项目，它们之所以能生成很精致的图片，很大程度上可能是因为使用了优秀的数据集，根据原作者的介绍，它们采用的数据集中图片画风很一致，质量都很高。而我们使用的数据集是台湾大学李宏毅老师在MLDS这门课程中提供的一个toy data set，我们浏览了一下，发现里面不仅有女性人物的画像，还混入了一定比例的男性人物画像，而且有的图片本身就画得很崩。这很大程度上会为对我们的generator产生干扰。

	![奇怪的数据集](https://raw.githubusercontent.com/Unispac/blogImageBed/master/人工智能/GAN/Animation%20Avatar%20Generation/img/29.png)

- 我们由于只有64x64的训练数据，所以也只能产生64x64的结果。借助stackGAN技术和更大尺寸的训练图片，我们可以生成分辨率更高的图片。