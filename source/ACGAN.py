import os 
import time
import tensorflow as tf
import numpy as np
import random
from tools import *
from resNet import *


layers = tf.contrib.layers
slim = tf.contrib.slim
Dropout = tf.layers.dropout


######## ACGAN 模型 ########
class ACGAN:
    modelName = "ACGAN"

    def __init__(self,sess,epoch,batchSize,codeSize,dataSource,\
        checkpointDir,resultDir,logDir,mode,height,width,Nchannel):
    ##### 初始化模型处理环境 #####
        #print("wocao ? ")
        #exit(0)
        self.sess=sess #绑定模型的会话上下文
        self.dataSource=dataSource #训练数据源
        self.checkpointDir=checkpointDir #保存检查点的目录
        self.resultDir=resultDir #保存结果的目录
        self.logDir=logDir #保存日志文件的目录
        
        self.epoch=epoch #训练轮数
        self.batchSize=64 #每一个batch的大小

        self.imageHeight=64  #图片高度
        self.imageWidth=64    #图片宽度
        self.imageChannel=3 #图片通道数

        self.codeSize=64  #图片编码向量的尺寸
        self.labelSize=23


        self.dIterNums=2 #判别器在每一个epoch中训练五轮
        self.gIterNums=1 #生成器在每一个epoch中训练一轮

        self.dropout = tf.placeholder(tf.float32)
        """
            更好的判别器可以给生成器更好的指引。有了GP，基本上解决了
            判别器太好导致生成器训练不动的问题，所以要充分利用判别器可以提前训练好的特性。
        """

        self.learning_rate = 0.0001 #Adam优化器的训练率
        self.beta1 = 0.5
        self.beta2 = 0.9   #两个参数
        self.la = 5
        
        self.numOfSamples = 25  #生成样例图片的数量
        
        if mode == "train" :  
            print("开始加载数据集！")
            self.imageData,self.lableData = loadImages(self.dataSource) 
            self.numOfBatches = len(self.imageData)//self.batchSize
            #self.numOfBatches = 512
            print("数据集加载成功!")
            print("numOfBatches : ",self.numOfBatches)
        """
            如果是训练模式，提前把训练集都加载进来。
            并且计算好这些数据需要总共打包多少个batches。
        """

   
    def discriminator(self,x,isTraining=True,reuse=True,update_collection=None):
    ##### 判别器 #####
        with tf.variable_scope("discriminator") as scope:
            
            if reuse:
                scope.reuse_variables()
    
            with tf.variable_scope('input_stage'):
                net = conv2(x, 4, 32, 2, scope='conv')
                net = lrelu(net, 0.2)


            res = net
            # The discriminator block part
            # block 1
            
            net = discriminator_block(net, 32, 3, 1, 'disblock_1')

            # block 2
            net = discriminator_block(net, 32, 3, 1, 'disblock_1_1')

            #def conv2d_sn(ibatch_input, kernel=3, output_channel=64, stride=1, scope='conv'):
            #net = conv2(net, 4, 64, 2, use_bias=False, scope='dis_conv_1')
            net = conv2d_sn(   net,  4 , 64, 2,  scope='dis_conv_1')

            # block 3
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_1')
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_2')
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_3')
            net = discriminator_block(net, 64, 3, 1, 'disblock_2_4')

            #net = conv2(net, 4, 128, 2, use_bias=False, scope='dis_conv_2')
            net = conv2d_sn(   net,  4 ,128, 2)
            net = lrelu(net, 0.2)

            # block 4
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_1')
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_2')
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_3')
            net = discriminator_block(net, 128, 3, 1, 'disblock_3_4')

            #net = conv2(net, 3, 256, 2, use_bias=False, scope='dis_conv_3')
            net = conv2d_sn(   net,  3, 256, 2, scope='dis_conv_3')
            net = lrelu(net, 0.2)


            net = discriminator_block(net, 256, 3, 1, 'disblock_4_1')
            net = discriminator_block(net, 256, 3, 1, 'disblock_4_2')
            net = discriminator_block(net, 256, 3, 1, 'disblock_4_3')
            net = discriminator_block(net, 256, 3, 1, 'disblock_4_4')


            #net = conv2(net, 3, 512, 2, use_bias=False, scope='dis_conv_4')
            net = conv2d_sn(   net,  3,512,2, scope='dis_conv_4')
            net = lrelu(net, 0.2)

            net = discriminator_block(net, 512, 3, 1, 'disblock_5_1')
            net = discriminator_block(net, 512, 3, 1, 'disblock_5_2')
            net = discriminator_block(net, 512, 3, 1, 'disblock_5_3')
            net = discriminator_block(net, 512, 3, 1, 'disblock_5_4')

            #net = conv2(net, 3, 1024, 2, use_bias=False, scope='dis_conv_5')
            net = conv2d_sn(net,  3, 1024, 2, scope='dis_conv_5')
            net = lrelu(net, 0.2)

            net = layers.flatten(net)

            net_class = lrelu(layers.fully_connected(net, 512, normalizer_fn=None, activation_fn=None))
            net_class = lrelu(layers.fully_connected(net_class, 256, normalizer_fn=None, activation_fn=None))
            net_class = layers.fully_connected(net_class, 23, normalizer_fn=None, activation_fn=None)

            net = lrelu(layers.fully_connected(net, 512, normalizer_fn=None, activation_fn=None))
            net = lrelu(layers.fully_connected(net, 256, normalizer_fn=None, activation_fn=None))
            net = layers.fully_connected(net, 1,  normalizer_fn=None, activation_fn=None)
        return net, net_class


       
    
    def generator(self,z,y,isTraining=True, reuse=False):
    ##### 生成器 #####
        with tf.variable_scope("generator") as scope:

            s = 64 # output image size [64]

            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            if reuse:
                scope.reuse_variables()
            
            noise_vector = tf.concat([z, y], axis=1)

            net_h0 = layers.fully_connected(
                noise_vector, 64*s8*s8,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=None
                )

            net_h0 = tf.layers.batch_normalization(net_h0, training=isTraining)
            net_h0 = tf.reshape(net_h0, [-1, s8, s8, 64])
            net = tf.nn.relu(net_h0)

            input_stage = net
  
            for i in range(16):
                name_scope = 'resblock_%d'%(i)
                net = residual_block(net, 64, 1, name_scope, train=isTraining)


            net =  tf.layers.batch_normalization(net, training=isTraining)
            net = tf.nn.relu(net)

            net = input_stage + net

            net = conv2(net, 3, 256, 1,scope='conv1')
            net = pixelShuffler(net, scale=2)
            net =  tf.layers.batch_normalization(net, training=isTraining)
            net = tf.nn.relu(net)

            net = conv2(net, 3, 256, 1, scope='conv2')
            net = pixelShuffler(net, scale=2)
            net =  tf.layers.batch_normalization(net, training=isTraining)
            net = tf.nn.relu(net)
            
            net = conv2(net, 3, 256, 1, scope='conv3')
            net = pixelShuffler(net, scale=2)
            net =  tf.layers.batch_normalization(net, training=isTraining)
            net = tf.nn.relu(net)
   

            net = conv2(net, 9, 3, 1, scope='conv4')

            net = tf.nn.tanh(net)
            
            return net

    def buildNet(self):
    ##### 实例化网络框架 #####
        np.random.seed(9487)
        random.seed(9487)
        tf.set_random_seed(9487)
        print("网络实例化:")
        imageSize = [self.imageHeight,self.imageWidth,self.imageChannel]
        bs = self.batchSize
        
        self.realImageInput = tf.placeholder(tf.float32, [bs]+imageSize,name='realImageInput') #真图输入接口
        imageFlip = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),self.realImageInput) 
        #以0.5的概率随机执行对图片的左右翻转。训练模型对脸部左右对称性的理解。
        angles = tf.random_uniform([bs], minval=-15.0 * np.pi / 180.0, maxval=15.0 * np.pi / 180.0, dtype=tf.float32, seed=9487)
        imageRotated = tf.contrib.image.rotate(imageFlip,angles,interpolation="NEAREST")
        #以均匀的概率将图片旋转-15~15度。训练模型对脸部轻微倾斜的理解。

        self.z = tf.placeholder(tf.float32, [bs,self.codeSize], name='Z') #噪音输入接口
        self.y = tf.placeholder(tf.float32, [bs,self.labelSize],name="L") #标签输入接口
        self.randomLable = tf.placeholder(tf.float32, [bs, self.labelSize], name="ramdomLable") #随机标签输入

        ##### Loss for Discrimiantor ######
       
        dRealFedality,dRealLable = self.discriminator(imageRotated,reuse=False,update_collection = None)
        dLossRealFedality = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dRealFedality, labels=tf.ones_like(dRealFedality)))
        # expect the fedality prediction is 1
        dLossRealLable = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=dRealLable, labels=self.y),axis=1))
        # expect the tag prediction is real tag : self.seq
        self.dLossReal = self.la * dLossRealFedality + dLossRealLable

        dFake = self.generator(self.z,self.randomLable)
        dFakeFedality, dFakeLable = self.discriminator(dFake,update_collection = "NO_OPS")
        dLossFakeFedality = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dFakeFedality, labels=tf.zeros_like(dFakeFedality)))
        self.dLossFake = self.la * dLossFakeFedality

        self.dLoss = self.dLossReal+self.dLossFake
        print("已构建 Loss for Discriminator")

        ##### Loss for Generator ######
        gFake = self.generator(self.z,self.randomLable,reuse=True)
        gFakeFedality, gFakeLable = self.discriminator(gFake,update_collection = "NO_OPS")
        gLossFedality = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gFakeFedality, labels=tf.ones_like(gFakeFedality)))
        gLossLable = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gFakeLable, labels=self.randomLable))
        self.gLoss = self.la * gLossFedality + gLossLable
        print("已构建 Loss for Generator")


        dVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'discriminator')
        gVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'generator')

        print("# size of dVars : ",len(dVars))
        print("# size of gVars : ",len(gVars))

        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.dOptimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=self.beta1,beta2=self.beta2)\
            .minimize(self.dLoss,var_list=dVars)
            self.gOptimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=self.beta1,beta2=self.beta2)\
                .minimize(self.gLoss,var_list=gVars)   #D和G的优化器
                
        print("已构建优化器")

        self.fakeImages = self.generator(self.z, self.y, isTraining=False,reuse=True) #用于单独的图片生成，在test的时候使用
        print("已构造预测器")

        #Summary : capture the key information for tensorBoard
        dLossSum = tf.summary.scalar("dLoss",self.dLoss)
        gLossSum = tf.summary.scalar("gLoss",self.gLoss)
        self.gSum = tf.summary.merge([gLossSum])
        self.dSum = tf.summary.merge([dLossSum])
        print("网络实例化成功!")

    def train(self):
        np.random.seed(9487)
        random.seed(9487)
        tf.set_random_seed(9487)
        
        print("开始配置训练环境!")

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver(max_to_keep=25)
        self.sumWriter = tf.summary.FileWriter(self.logDir+"/"+self.modelName,self.sess.graph)

        self.testZ=np.random.normal(0, np.exp(-1 / np.pi), size=(self.batchSize, self.codeSize))
        self.testLable=np.zeros((self.batchSize,self.labelSize))

        # blue hair, blue eye.
        for i in range(5):
            self.testLable[0 + i][8] = 1
            self.testLable[0 + i][22] = 1
        # white hair, green eye.
        for i in range(5):
            self.testLable[5 + i][1] = 1
            self.testLable[5 + i][19] = 1
        # pink hair, red eye.
        for i in range(5):
            self.testLable[10 + i][7] = 1
            self.testLable[10 + i][21] = 1
        # green hair, blue eye.
        for i in range(5):
            self.testLable[15 + i][4] = 1
            self.testLable[15 + i][22] = 1
        # aquahair, red eye.
        for i in range(5):
            self.testLable[20 + i][2] = 1
            self.testLable[20 + i][21] = 1

        loadSuccess,checkPointCounter = self.loadModel(self.checkpointDir)
        if loadSuccess:
            startEpoch = checkPointCounter
            counter = checkPointCounter
            print("加载成功")
            print("生成模型结果预览")
            self.visualization(counter-1,self.testZ)
        else:
            startEpoch = 0
            counter = 0
            print("没有找到检查点，模型从初始化状态开始")
        
        

        
        gLoss = 0.0
        print("训练开始！～～～～～～～～～～～～～～～～～～～～～～～～")
        timeStart = time.time()

        dataSize = len(self.imageData)

        for epoch in range(startEpoch,self.epoch):
            for bid in range(0,self.numOfBatches):
                
                imageBatch = np.asarray(self.imageData[bid*self.batchSize:(bid+1)*self.batchSize]).astype(np.float32)
                lableBatch = np.asarray(self.lableData[bid*self.batchSize:(bid+1)*self.batchSize]).astype(np.float32)
                #wrongImageBatch = np.asarray(self.imageData[random.sample(range(dataSize),self.batchSize)]).astype(np.float32)
                wrongLableBatch = np.asarray(self.lableData[random.sample(range(dataSize),self.batchSize)]).astype(np.float32)
                zBatch = np.random.normal(0,np.exp(-1.0/np.pi),[self.batchSize,self.codeSize]).astype(np.float32)


                for cntD in range(self.dIterNums):
                    opResult, summaryString, dLoss = self.sess.run([self.dOptimizer,self.dSum,self.dLoss],
                    feed_dict={self.realImageInput:imageBatch,self.z:zBatch,self.y:lableBatch,self.randomLable:wrongLableBatch,self.dropout:0.3})
                    self.sumWriter.add_summary(summaryString,counter)

                zBatch = np.random.normal(0,np.exp(-1.0/np.pi),[self.batchSize,self.codeSize]).astype(np.float32)
                
                for cntG in range(self.gIterNums):
                    opResult, summaryString, gLoss = self.sess.run([self.gOptimizer,self.gSum,self.gLoss],
                    feed_dict={self.z:zBatch,self.randomLable:wrongLableBatch,self.realImageInput:imageBatch,self.dropout:0.0})
                    self.sumWriter.add_summary(summaryString,counter)

                print("Epoch: [%4d/%4d] [%4d/%4d] time: %4.4f seconds, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, self.epoch, bid, self.numOfBatches, time.time() - timeStart, dLoss, gLoss), end='\r')
            counter +=1
            print()
            self.saveModel(self.checkpointDir,counter)
            self.visualization(epoch,self.testZ)


    
    def visualization(self,epoch,z_sample):   #传入一个编码样本，产生一个结果样本。
        cntOfSample = min(self.numOfSamples,len(z_sample))
        windowDim = int(np.floor(np.sqrt(cntOfSample)))
        Samples = self.sess.run(self.fakeImages,feed_dict={self.z: z_sample,self.y: self.testLable,self.dropout:0.0})  #从网络运算中得到结果
        resultPath = checkFolder(self.resultDir+ '/' + self.modelName) + '_epoch%03d' % epoch + '.png' #确定保存路径
        saveImages(Samples[:windowDim*windowDim,:,:,:],[windowDim,windowDim],resultPath) #保存
    
    def buildForInfer(self):
        bs = self.batchSize
        self.z = tf.placeholder(tf.float32, [bs,self.codeSize], name='Z') #噪音输入接口
        self.y = tf.placeholder(tf.float32, [bs,self.labelSize],name="L") #标签输入接口
        self.fakeImages = self.generator(self.z, self.y, isTraining=False,reuse=False) #用于单独的图片生成，在test的时候使用
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(max_to_keep=25)
        loadSuccess,checkPointCounter = self.loadModel(self.checkpointDir)
        if loadSuccess:
            print("已构造预测器")
        else:
            print("没有可用的模型!")

    def infer(self,targetLable,feature):
        z=np.random.normal(0, np.exp(-1 / np.pi), size=(self.batchSize, self.codeSize))
        cntOfSample = 25
        windowDim = 5
        Samples = self.sess.run(self.fakeImages,feed_dict={self.z: z , self.y: targetLable})  #从网络运算中得到结果
        resultPath = checkFolder('./samples/') + feature + '.png' #确定保存路径
        saveImages(Samples[:windowDim*windowDim,:,:,:],[windowDim,windowDim],resultPath) #保存
      





    def saveModel(self,checkpointDir,step):           #保存模型
        checkpointDir = os.path.join(checkpointDir,self.modelName)
        if not os.path.exists(checkpointDir):
            os.makedirs(checkpointDir)
        print("新的模型将会被保存 : ",checkpointDir)
        self.saver.save(self.sess,os.path.join(checkpointDir,self.modelName+".model"),global_step=step)
    
    def loadModel(self,checkpointDir):               #加载模型
        checkpointDir = os.path.join(checkpointDir,self.modelName)
        print("模型将会被加载 : ",checkpointDir)
        ckpt = tf.train.get_checkpoint_state(checkpointDir)  #从模型保存的目录下拉取模型信息
        if ckpt and ckpt.model_checkpoint_path:
            modelName = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,os.path.join(checkpointDir,modelName))
            print("MODEL NAME : ",modelName)
            st = modelName.index("-")
            counter = int(modelName[st+1:])
            print("模型加载成功 : ",modelName)
            return True,counter
        else :
            print("没有找到检查点")
            return False,0

