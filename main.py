# -*- coding: utf-8 -*-
import os
import argparse
import time
import datetime
import sys
import platform

import keras
from keras.layers import *
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper
from path import MODEL_PATH, DATA_PATH
import pandas as pd
import numpy as np

from WangyiUtilOnFlyai import *
import hyperparameter as hp
from net import Net
from data_helper import load_dict, load_labeldict, get_batches, read_data


'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
args = parser.parse_args()

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
        pass

time_now = time.strftime("%Y-%m-%d %H-%M-%S")
time_now_csv = str(time_now) + '.csv'
time_now_log = str(time_now) + '.txt'
time_now_plt = str(time_now) + '.png'

if not os.path.exists(os.path.join(sys.path[0], 'data', 'output')):
    os.makedirs(os.path.join(sys.path[0], 'data', 'output'))
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def download_data(self):
        # 下载数据
        data_helper = DataHelper()
        data_helper.download_from_ids("MedicalClass")
        # print('=*=数据下载完成=*=')

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        # 加载数据
        # self.data = pd.read_csv(os.path.join(DATA_PATH, 'MedicalClass/train.csv'))
        # file_unique = self.data['label'].unique()
        # print('train data label 分类：', len(file_unique))
        # 划分训练集、测试集 https://blog.csdn.net/datascientist_chen/article/details/79024020
        # self.train_data, self.valid_data = train_test_split(self.data, test_size=0.1, random_state=6, shuffle=True)
        self.text2id, _ = load_dict(os.path.join(DATA_PATH, 'MedicalClass/words_fr.dict'))
        self.label2id, _ = load_labeldict(os.path.join(DATA_PATH, 'MedicalClass/label.dict'))
        # self.train_text, self.train_label = read_data(self.train_data, self.text2id, self.label2id)
        # self.val_text, self.val_label = read_data(self.valid_data, self.text2id, self.label2id)
        # print('=*=数据处理完成=*=')

    def train(self):
        '''
        0. 初始化参数
        '''
        sys.stdout = Logger(os.path.join(sys.path[0], 'data', 'output', time_now_log))  # 保存log到data/output/
        # 读取数据
        hp.num_classes = len(get_nubclass_from_csv())
        d_batchSize = dynamicBatchSize(hp.num_classes)
        dataset_wangyi = DatasetByWangyi(get_nubclass_from_csv())
        dataset_wangyi.set_Batch_Size(train_size=d_batchSize.getSizebyAcc(0), val_size=hp.val_batch_size)
        draw_plt = drawMatplotlib()
        draw_plt.set_path(path=os.path.join(sys.path[0], 'data', 'output', time_now_plt))
        best_score = bestScore()
        if platform.system() == 'Windows':
            set_num_workers = 1
            print('Windows , use num_workers = 1')
        else:
            set_num_workers = 0
            print('not Windows , use num_workers = 0')


        rnn_unit_1 = 128    # RNN层包含cell个数
        embed_dim = 64      # 嵌入层大小
        class_num = len(self.label2id)
        num_word = len(self.text2id)
        batch_size = args.BATCH
        MAX_SQUES_LEN = 68  # 最大句长

        '''
        0.1 / 构建cnn
        '''
        text_input = Input(shape=(MAX_SQUES_LEN,), dtype='int32')
        embedden_seq = Embedding(input_dim=num_word, output_dim=embed_dim, input_length=MAX_SQUES_LEN)(text_input)
        BN1 = BatchNormalization()(embedden_seq)
        bGRU1 = Bidirectional(GRU(rnn_unit_1, activation='selu', return_sequences=True,
                                  implementation=1), merge_mode='concat')(BN1)
        bGRU2 = Bidirectional(GRU(rnn_unit_1, activation='selu', return_sequences=True,
                                  implementation=1), merge_mode='concat')(bGRU1)

        drop = Dropout(0.5)(bGRU2)
        avgP = GlobalAveragePooling1D()(drop)
        maxP = GlobalMaxPooling1D()(drop)

        conc = concatenate([avgP, maxP])

        pred = Dense(class_num, activation='softmax')(conc)
        k_model = keras.Model(text_input, pred)
        k_model.summary()
        k_model.compile(optimizer=RMSprop(lr=0.0005), loss='categorical_crossentropy', metrics=['acc'])


        for epoch in range(args.EPOCHS):
            time_1 = time.time()
            train_batch = dataset_wangyi.get_Next_Batch() # , val_batch
            print('epoch',epoch)
            '''
            2.1 train
            '''
            time_train = time.time()
            train_text, train_label = read_data(train_batch, self.text2id, self.label2id)
            x_train,y_train = get_batches(train_text, train_label, text_padding=self.text2id['_pad_'])
            history = k_model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, verbose=0)
            train_acc = history.history['acc'][0]
            train_loss = history.history['loss'][0]
            print('train acc : %.4f, loss : %.4f, take time:%.1f' % (train_acc, train_loss, time.time() - time_train))

            '''
            2.2 validate
            '''
            # time_val = time.time()
            # val_acc = 0.0
            # val_loss = 0.0
            #
            # x_val, y_val = get_val_batch(self.val_text, self.val_label,
            #                              batch_size=1024, text_padding=self.text2id['_pad_'])
            # if batch_i % 100 == 0:
            #     score = k_model.evaluate(np.array(x_val), np.array(y_val), batch_size=args.BATCH,verbose=0)
            #     val_acc = score[1]
            #     if val_acc > best_score:
            #         best_score = val_acc
            #         k_model.save(os.path.join(MODEL_PATH, 'model.h5'))
            #     print('best acc:', best_score)
            #     print('val acc : %.4f, loss : %.4f, take time:%.1f' % (
            #     val_acc, val_loss, time.time() - time_val))

            cost_time = time.time() - time_1
            need_time_to_end = datetime.timedelta(
                seconds=(args.EPOCHS - epoch - 1) * int(cost_time))
            print('耗时：%d秒,预估还需' % (cost_time), need_time_to_end)


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()

    exit(0)