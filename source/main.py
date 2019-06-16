import os
from ACGAN import ACGAN
from tools import checkFolder
import tensorflow as tf
import argparse
import numpy as np

def parse_args():
    note = "ACGAN Frame Constructed With Tensorflow"
    parser = argparse.ArgumentParser(description=note)
    parser.add_argument("--epoch",type=int,default=400,help="训练轮数")
    parser.add_argument("--batchSize",type=int,default=64,help="batch的大小")
    parser.add_argument("--codeSize",type=int,default=62,help="输入编码向量的维度")
    parser.add_argument("--checkpointDir",type=str,default="./checkpoint",help="检查点保存目录")
    parser.add_argument("--resultDir",type=str,default="./result",help="训练过程中，中间生成结果的目录")
    parser.add_argument("--logDir",type=str,default="./log",help="训练日志目录")
    parser.add_argument("--mode",type=str,default="train",help="模式： train / infer")
    parser.add_argument("--dataSource",type=str,default='./extra_data/images/',help="训练集路径")
    args = parser.parse_args()
    checkFolder(args.checkpointDir)
    checkFolder(args.resultDir)
    checkFolder(args.logDir)
    assert args.epoch>=1
    assert args.batchSize>=1
    assert args.codeSize>=1
    return args

def main():
    args=parse_args()
    if args is None :
        print("命令行参数错误！")
        exit(0)
    """
        def __init__(self,sess,epoch,batchSize,codeSize,dataSource,\
        checkpointDir,resultDir,logDir,mode,height,width,Nchannel):
    """
    with tf.Session() as sess :
        myGAN = ACGAN(sess,args.epoch,args.batchSize,args.codeSize,\
            args.dataSource,args.checkpointDir,args.resultDir,args.logDir,args.mode,\
                64,64,3)
        if myGAN is None:
            print("创建GAN网络失败")
            exit(0)
        

        if args.mode=='train' :
            myGAN.buildNet()
            print("进入训练模式")
            myGAN.train()
            print("Done")
        elif args.mode=='infer' :
            myGAN.buildForInfer()
            tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 
                        'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair','gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
                        'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
            hair_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 
                        'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair']
            eye_dict = [ 'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
                        'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
            hairStyle = str()
            eyeStyle = str()
            """
            mark = "yes"
            while mark=="yes" :
                print("######  Allowed Hair Style  ######### ")
                print(hair_dict)
                print("#############################")
                print()
                hairStyle=input("Choose your hair style : ")
                print()

                print("###### Allowed Eyes Style ##########")
                print(eye_dict)
                print("##############################")
                print()
                eyeStyle=input("Choose your eyes style : ")
                print()

                tag = np.zeros((64,23))
                feature = hairStyle+" AND "+ eyeStyle
                for j in range(25):
                    for i in range(len(tag_dict)):
                        if tag_dict[i] in feature:
                            tag[j][i] = 1
                myGAN.infer(tag,feature)

                print("Done!")
                mark=input("continue ? yes/no : ")
           """
            for hairStyle in hair_dict:
                for eyeStyle in eye_dict:
                    tag = np.zeros((64,23))
                    feature = hairStyle+" AND "+ eyeStyle
                    for j in range(25):
                        for i in range(len(tag_dict)):
                            if tag_dict[i] in feature:
                                tag[j][i] = 1
                    myGAN.infer(tag,feature)
                    print("Generate : "+feature)


main()

