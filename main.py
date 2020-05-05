# -*- coding: utf-8 -*-
import os
import argparse
import time
import datetime
import sys
import platform

import keras
from keras.layers import *
from keras.optimizers import RMSprop,Adam
from sklearn.model_selection import train_test_split
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper
from path import MODEL_PATH, DATA_PATH
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from flyai.utils.log_helper import train_log
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
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
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
        # hp.val_batch_size = [16] * hp.num_classes
        dataset_wangyi.set_Batch_Size(train_size=d_batchSize.getSizebyAcc(0), val_size=hp.val_batch_size_5000)
        draw_plt = drawMatplotlib()
        draw_plt.set_path(path=os.path.join(sys.path[0], 'data', 'output', time_now_plt))
        best_score = bestScore()
        if platform.system() == 'Windows':
            set_num_workers = 1
            print('Windows , use num_workers = 1')
        else:
            set_num_workers = 0
            print('not Windows , use num_workers = 0')

        '''
        0.1 / 构建cnn
        '''
        model_net = Net(num_classes =  get_nubclass_from_csv(),label2id=self.label2id , text2id=self.text2id)
        k_model = model_net.get_Model()
        k_model.summary()
        k_model.compile(optimizer=Adam(lr=3e-5), loss='categorical_crossentropy', metrics=['accuracy'])

        predict_csv = {}
        # predict_csv['truth'] = y_val
        for epoch in range(args.EPOCHS):
            time_1 = time.time()
            draw_plt.epoch_train_x.append(epoch + 1)


            train_batch,val_batch = dataset_wangyi.get_Next_Batch() # , val_batch
            draw_plt.train_length_list.append(len(train_batch))
            # 打印步骤和训练集/测试集的量
            cur_step = str(epoch + 1) + "/" + str(args.EPOCHS)
            print('------------------')
            print('■' + cur_step, ':train %d,val %d ' % (len(train_batch), len(val_batch)))
            '''
            2.1 train
            '''
            time_train = time.time()
            train_text_x1, train_text_x2,train_label = read_data_v2(train_batch, self.text2id, self.label2id)
            val_text_x1, val_text_x2, val_label = read_data_v2(val_batch, self.text2id, self.label2id)

            history = k_model.fit([train_text_x1, train_text_x2], np.array(train_label),
                                  # validation_data=(np.array(x_val), np.array(y_val)),
                                  batch_size=args.BATCH, verbose=2)
            train_acc = history.history['acc'][0]
            train_loss = history.history['loss'][0]
            # val_acc = history.history['val_acc'][0]
            # val_loss = history.history['val_loss'][0]
            # print('train acc : %.4f, loss : %.4f, take time:%.1f' % (train_acc, train_loss, time.time() - time_train))
            draw_plt.train_acc_list.append(train_acc)
            draw_plt.train_loss_list.append(train_loss)
            # draw_plt.val_acc_list.append(val_acc)
            # draw_plt.val_loss_list.append(val_loss)
            # draw_plt.learn_rate_list_y.append(optimizer.state_dict()['param_groups'][0]['lr'])

            # draw_plt.train_loss_list.extend([train_loss] * len(train_data_loader))

            # draw_plt.val_loss_list.extend([val_loss] * len(valid_data_loader))
            # draw_plt.f1_score.append(f1_weighted)
            '''
            2.2 validate
            '''
            val_acc = 0.0
            val_loss = 0.0
            prediction_total_validate = []  # 预测4403，shape = (4403,)
            time_val = time.time()

            val_history = k_model.predict([val_text_x1, val_text_x2])
            y_predict_numpy = np.argmax(np.array(val_history), axis=1)
            y_val_numpy = np.argmax(np.array(val_label), axis=1)

            f1_weighted = f1_score(y_val_numpy, y_predict_numpy, average='weighted')
            # 求出每类wrong acc(to acc1.0)，用来设置后续的train batch
            tmp_wrong = np.subtract(y_val_numpy, np.array(y_predict_numpy))
            val_acc = np.sum((tmp_wrong == 0) / tmp_wrong.shape[0])
            wrong_acc = []
            for j in range(hp.num_classes):
                index = y_val_numpy == j  # 索引0类、1类、、、以此类推
                wrong = np.sum((tmp_wrong[index] != 0) / tmp_wrong.shape[0])
                wrong_acc.append(round(wrong, 4))

            # predict_csv['epoch_' + str(epoch + 1)] = y_predict_numpy
            # predict_file = DataFrame(predict_csv)  # 将字典转换成为数据框
            # predict_file.to_csv(os.path.join(sys.path[0], 'data', 'output', time_now_csv))

            print('val acc : %.4f, loss : %.4f, f1_score : %.4f, take time:%.1f' % (
            val_acc, val_loss, f1_weighted, time.time() - time_val))
            draw_plt.val_acc_list.append(val_acc)
            draw_plt.val_loss_list.append(val_acc)
            draw_plt.f1_score.append(f1_weighted)

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

            '''
            3/ 保存最佳模型model
            '''

            # save best acc
            if best_score.judge_and_save(val_acc, val_loss, epoch, f1_score=f1_weighted):
                k_model.save(os.path.join(MODEL_PATH, 'model.h5'))
                print('save best model')
            best_score.print_best_now()
            '''
            4/ 调整学习率和优化模型
            '''

            '''
           4.1/ 调整train batch
           '''
            # train acc > 99% (loss < 0.04) ，启动two-phrase training，冻结特征层然后只训练全连接层

            dataset_wangyi.set_Batch_Size(train_size=d_batchSize.getSizebyAcc(val_acc, wrong_acc=wrong_acc),
                                          val_size=hp.val_batch_size_5000)

            '''
            5、控制台输出，和matplotlib输出
            '''
            # 调用系统打印日志函数，这样在线上可看到训练和校验准确率和损失的实时变化曲线
            train_log(train_loss=round(train_loss, 4),
                      train_acc=round(train_acc , 4),
                      val_loss=round(val_loss, 4),
                      val_acc=round(val_acc, 4)
                      )
            sys.stdout.flush()
            # 输出plot
            if platform.system() == 'Windows':
                draw_plt.showPlt(best_score_class=best_score)
            # draw_plt.showPlt(best_score_by_acc=best_score_by_acc, best_score_by_loss=best_score_by_loss, best_epoch=best_epoch+1)
            '''
            6、耗时统计
            '''
            cost_time = time.time() - time_1
            need_time_to_end = datetime.timedelta(
                seconds=(args.EPOCHS - epoch - 1) * int(cost_time))
            print('耗时：%d秒,预估还需' % (cost_time), need_time_to_end)

        if os.path.exists(MODEL_PATH):
            best_score.print_best_final()
        else:
            print('未达到save best acc的条件，已保存最后一次运行的model')
            k_model.save(os.path.join(MODEL_PATH, 'model.h5'))




if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()

    exit(0)