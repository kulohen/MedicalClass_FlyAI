#!/usr/bin/env python
# coding:utf-8
"""
Name : WangyiUtilOnFlyai.py
Author  : 莫须有的嚣张
Contect : 291255700
Time    : 2019/7/28 上午9:42
Desc:
"""
import os
from time import clock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from flyai.source.base import DATA_PATH
from torch import optim as optmzs
from torchvision import transforms

from data_helper import *
import net
from hyperparameter import random_per_epoch, reduce_lr_per_epochs, scale_num, img_size, num_classes, train_vs_val
from path import DATA_PATH


DATA_ID = 'MedicalClass'

# 保留的图片中心的比例，如80%，写0.8，同时在prediction.py里要做修改
center_scale = 0.85


def get_1_x_data(image_path):
    img = Image.open(os.path.join(DATA_PATH, DATA_ID, image_path))

    return img


def get_1_y_data(label):

    return label


lr_level = [
    3e-4,
    0.0001,
    # 3e-5,
    # 1e-5,
]

optimizer_level = [
    'sgd',
    'rmsprop',
    'adagrad',
    # 'adadelta',
    'adam',
    # 'adamax'
]

optimizer_name = {
    'sgd': optmzs.SGD,
    'rmsprop': optmzs.RMSprop,
    'adagrad': optmzs.Adagrad,
    # 'adadelta': optmzs.Adadelta,
    'adam': optmzs.Adam,
    # 'adamax': optmzs.Adamax
}


class OptimizerByWangyi():
    def __init__(self):
        self.optimizer_iterator = 0
        self.lr_iterator = 0
        self.pationce_count = 0
        self.lr_level = 0  # 在训练时判断后+1 使用
        self.now_optimizer = None
        self.now_opt_name = None  # 当前的优化器
        self.now_opt_lr = None  # 当前的学习率
        self.now_opt_name_lv = -1  # 当前的优化器索引
        self.now_opt_lr_lv = -1  # 当前的学习率索引

    def get_create_optimizer(self, name, lr_num):
        if name is None or lr_num <= 0:
            raise ValueError('请指定正确的优化器/学习率')

        self.now_optimizer = optimizer_name[name]
        self.now_opt_name = name
        self.now_opt_lr = lr_num
        self.now_opt_name_lv = optimizer_level.index(name)
        self.now_opt_lr_lv = lr_level.index(lr_num)
        print('采用了优化器：', name, '--学习率:', lr_num)
        return self.now_optimizer, lr_num

    def get_next(self, optimzer=None, lr=None):

        if optimzer is not None:
            name_1 = optimzer
        else:
            name_1 = optimizer_level[self.optimizer_iterator]

        if lr is not None:
            lr_1 = lr
        else:
            lr_1 = lr_level[self.lr_iterator]

        x = self.get_create_optimizer(name_1, lr_1)

        self.lr_iterator = (self.lr_iterator + 1) % len(lr_level)
        if self.lr_iterator == 0:
            self.optimizer_iterator = (self.optimizer_iterator + 1) % len(optimizer_level)
        return x

    def get_random_opt(self):
        name = np.random.randint(0, len(optimizer_name))
        lr = np.random.randint(0, len(lr_level))
        print('启动随机de学习率')
        # self.now_opt_name = name
        # self.now_opt_lr = lr
        return self.get_create_optimizer(optimizer_level[name], lr_level[lr])

    def reduce_lr_by_loss_and_epoch(self, get_loss, get_epoch):
        tmp_opt = None

        # 多少个epochs后启动随机学习率，对应Warmup Scheduler的实现
        if get_epoch % random_per_epoch == random_per_epoch - 1:
            print('经过%d epochs：' % random_per_epoch)
            tmp_opt = self.get_random_opt()

        # 每此学习率都能规律下降，例如1e-3 到 1e-4 到1e-5

        if get_epoch < random_per_epoch:
            pass
        elif (get_epoch % random_per_epoch) % reduce_lr_per_epochs == reduce_lr_per_epochs - 1:
            # 降低学习率
            self.now_opt_lr_lv = (self.now_opt_lr_lv + 1) % len(lr_level)  # lr_level的范围
            print('经过%d epochs降低学习率' % reduce_lr_per_epochs)
            tmp_opt = self.get_create_optimizer(self.now_opt_name, lr_level[self.now_opt_lr_lv])

        # 调整学习率，且只执行一次
        if get_loss < 0.4 and self.lr_level < 1:
            print('train loss 低于 %.1f' % 0.4)
            tmp_opt = self.get_create_optimizer(name='adagrad', lr_num=1e-4)
            self.lr_level = 1

        elif get_loss < 0.1 and self.lr_level < 2:
            print('train loss 低于 %.1f' % 0.1)
            tmp_opt = self.get_create_optimizer(name='sgd', lr_num=3e-4)
            self.lr_level = 2

        return tmp_opt


class dynamicBatchSize():
    def __init__(self, num_class):
        self.n = num_class
        self.now_batch = [1] * num_class

    def getSizebyAcc(self, acc, wrong_acc=None):

        assert 0 <= acc <= 1
        if acc < 0.90:
            self.now_batch = [32] * self.n
            # self.now_batch = [128] * self.n
        elif acc < 0.99:
            self.now_batch = [32] * self.n
        elif acc < 0.995:
            # self.now_batch = super_batch
            self.now_batch = [8] * self.n

        elif acc <= 1:
             self.now_batch= [4] * self.n

        if wrong_acc is not None and acc >= 0.90:
            wrong_acc_dict = {}
            for i, value in enumerate(wrong_acc):
                self.now_batch[i] = int(self.now_batch[i] * (value * 1000) + 1)
                wrong_acc_dict[i] = value
            # wrong_acc_dict = {i:value for i, value in enumerate(wrong_acc)}
            # wrong_acc_dict= {}
            # for i, value in enumerate(wrong_acc):
            #     wrong_acc_dict[i] = value
            print('val acc:%.4f,wrong acc：' % acc, wrong_acc_dict)

        return self.now_batch


def get_sliceCSVbyClassify_V4(label='label', classify_count=21, split=0.8):
    # 2020-4-24 classify_count 输入label的list

    # 读取数据
    tmp_a = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
    tmp_b = tmp_a.sample(frac=1)

    label_list = tmp_b['label'].values
    title_list = tmp_b['title'].values
    text_list = tmp_b['text'].values

    # 划分训练集和校验集
    all_size = len(label_list)

    # step 2 : 筛选 csv

    list_path_train, list_path_test = [], []
    # for epoch in range(classify_count):
    for epoch,value in enumerate(classify_count):
        # 根据label来筛选
        index = tmp_b[label] == value
        tmp_c = tmp_b[index]
        cut_length = int(len(tmp_c) * split)
        a = tmp_c[: cut_length]
        b = tmp_c[cut_length:]

        path_train = 'wangyi-train-classfy-' + str(epoch) + '.csv'
        a.to_csv(os.path.join(DATA_PATH, DATA_ID, path_train), index=False)
        list_path_train.append(path_train)

        path_test = 'wangyi-test-classfy-' + str(epoch) + '.csv'
        b.to_csv(os.path.join(DATA_PATH, DATA_ID, path_test), index=False)
        list_path_test.append(path_test)

        print('classfy-', epoch, ' : train and test.csv save OK!')

    return list_path_train, list_path_test


def getDatasetListByClassfy_V5(classify_count=21):
    # 2020-3-21 flyai改版本了，这是为了适应

    xx, yy = get_sliceCSVbyClassify_V4(classify_count=classify_count, split=train_vs_val)
    list_train = []
    list_val = []
    train_total_list = []
    val_total_list = []
    for epoch in range(len(classify_count)):
        time_0 = clock()
        # 读取数据
        tmp_train = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, xx[epoch]))
        tmp_val = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, yy[epoch]))

        # dataset = Lib(source=readCustomCsv_V3(xx[epoch], yy[epoch]), epochs=10)
        list_train.append(tmp_train)
        list_val.append(tmp_val)
        train_total_list.append(len(tmp_train))
        val_total_list.append(len(tmp_val))
        print('class-', epoch, ' 的data 建立成功, 耗时：%.1f 秒' % (clock() - time_0), '; train_length:',
              len(tmp_train), '; val_length:', len(tmp_val))

    return list_train, list_val, train_total_list, val_total_list


class DatasetByWangyi():
    def __init__(self, n):
        self.num_classes = n # n是label unique 的list
        self.dropout = 0.5

        time_0 = clock()
        self.train_pd_list, self.val_pd_list, self.train_total_list, self.val_total_list = getDatasetListByClassfy_V5(
            classify_count=n)
        self.train_step_list = [0] * len(n)
        self.val_step_list = [0] * len(n)

        print('全部分类的flyai dataset 建立成功, 耗时：%.1f 秒' % (clock() - time_0))

        self.train_batch_List = []
        self.val_batch_size = []
        self.data_dict_x = {}
        self.data_dict_y = {}
        self.read_All_to_Memory()


    def set_Batch_Size(self, train_size, val_size):
        self.train_batch_List = train_size
        self.val_batch_size = val_size

    def read_All_to_Memory(self, multitask=False):
        '''
        # 读取数据
        time_1 = clock()
        tmp_a = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
        label_list = tmp_a['label'].values
        title_list = tmp_a['title'].values
        text_list = tmp_a['text'].values
        # 生成x,y，以dict形式存在
        if multitask:
            pass
        else:
            for i, image in enumerate(image_path_list):
                self.data_dict_x[image] = get_1_x_data(image)
                self.data_dict_y[image] = get_1_y_data(label_list[i])
        print('全部图片加载到内存, 耗时：%.1f 秒' % (clock() - time_1))
        return self.data_dict_x, self.data_dict_y

        '''
        pass

    def get_Next_Batch(self):
        # 平衡输出45类数据
        train = self.get_Next_Train_Batch_fromMemory()
        # val = self.get_Next_Val_Batch_fromMemory()
        return train # ,val

    def get_Next_Train_Batch_fromMemory(self):
        # 平衡输出45类数据
        value_list = []
        # 改成生成list
        for iters in range(len(self.num_classes)):
            for i in range(self.train_batch_List[iters]):
                # 读取数据
                if len(self.train_pd_list[iters]['label'])==0:
                    continue
                # value = self.train_pd_list[iters]['label'].values[self.train_step_list[iters]]
                value = self.train_pd_list[iters].values[self.train_step_list[iters]]
                # print(value)
                value_list.append(value)
                # 修改游标
                self.train_step_list[iters] += 1
                self.train_step_list[iters] = self.train_step_list[iters] % self.train_total_list[iters]

        return value_list

    def get_Next_Val_Batch_fromMemory(self):
        value_list = []
        for iters in range(len(self.num_classes)):
            for i in range(self.val_batch_size[iters]):
                # 读取数据
                if len(self.val_pd_list[iters]['label'])==0:
                    continue
                value = self.val_pd_list[iters]['label'].values[self.val_step_list[iters]]
                value_list.append(value)
                # 修改游标
                self.val_step_list[iters] += 1
                self.val_step_list[iters] = self.val_step_list[iters] % self.val_total_list[iters]

        return value_list

def label_smoothing(inputs, epsilon=0.1):
    return (1.0 - epsilon) * inputs + epsilon / num_classes


class drawMatplotlib():
    def __init__(self):
        self.path = None
        self.train_acc_list = []  # 收集用来matplotlib
        self.val_acc_list = []  # 收集用来matplotlib
        self.f1_score = []  # 收集用来matplotlib
        self.train_loss_list = []  # 收集用来matplotlib
        self.train_loss_list_y = []  # 收集用来matplotlib
        self.val_loss_list = []  # 收集用来matplotlib
        self.learn_rate_list_y = []  # 收集用来matplotlib
        self.step_train_x = []  # 收集用来matplotlib
        self.step_val_x = []
        self.epoch_train_x = []
        self.train_length_list = []  # 收集用来matplotlib ，训练集每个epoch里的量（length）

    def set_path(self, path):
        self.path = path

    def showPlt(self, best_score_by_acc=0, best_score_by_loss=999, best_epoch=1, best_score_class=None):
        if best_score_class is not None:
            best_score_by_acc = best_score_class.best_score_by_acc
            best_score_by_loss = best_score_class.best_score_by_loss
            best_epoch = best_score_class.best_epoch
            best_f1_score = best_score_class.best_f1_score
        plt.clf()
        plt.figure('train and validate Accuracy')
        plt.figure(figsize=(12 + int(len(self.epoch_train_x) / 10), 8))
        plt.subplot(2, 2, 1)
        if best_score_by_acc != 0:
            plt.title('best: acc:%.4f, loss:%.4f, epoch:%d, f1_score:%.4f' % (
            best_score_by_acc, best_score_by_loss, best_epoch + 1, best_f1_score))
        else:
            plt.title('do not satisfy best_score condition')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(self.epoch_train_x, self.train_acc_list, label='train_acc')
        plt.plot(self.epoch_train_x, self.val_acc_list, label='val_acc')
        plt.plot(self.epoch_train_x, self.f1_score, label='f1_score')
        plt.legend(['train_acc', 'val_acc', 'f1_score'], loc='upper left')

        plt.subplot(2, 2, 2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(self.step_train_x, self.train_loss_list, label='train_loss')
        # 不计算val loss
        # plt.plot(self.step_val_x, self.val_loss_list, label='val_loss')
        # plt.plot(self.step_train_x, self.train_loss_list_y, label='per_train_loss')
        plt.legend(['train_loss', 'val_loss', 'per_train_loss'], loc='upper left')

        plt.subplot(2, 2, 3)
        plt.xlabel('Epochs')
        plt.ylabel('learn rate')
        plt.plot(self.step_train_x, self.learn_rate_list_y, label='learn_rate')
        plt.legend(['learn_rate'], loc='upper left')

        plt.subplot(2, 2, 4)
        plt.xlabel('Epochs')
        plt.ylabel('train length')
        plt.bar(self.epoch_train_x, self.train_length_list, label='train_length', align='center', width=0.8)
        plt.legend(['train_length'], loc='upper left')

        if self.path is not None:
            plt.savefig(self.path)
        plt.show()  # You must call plt.show() to make graphics appear.


class bestScore():
    def __init__(self):
        self.best_score_by_acc = 0.
        self.best_score_by_loss = 999.
        self.best_epoch = 0
        self.best_f1_score = 0
        # 保存最佳model的精准度，比如1%的准确范围写1,若2%的保存范围写2
        self.save_boundary = 0.1

    def judge_and_save(self, val_acc, val_loss, epoch, f1_score=None):

        if val_acc > 0.99:
            if f1_score is not None:
                if f1_score >= self.best_f1_score:
                    self.best_score_by_acc = val_acc
                    self.best_score_by_loss = val_loss
                    self.best_epoch = epoch
                    self.best_f1_score = f1_score
                    return True
            elif round(self.best_score_by_acc / self.save_boundary, 2) < round(
                    val_acc / self.save_boundary, 2):
                self.best_score_by_acc = val_acc
                self.best_score_by_loss = val_loss
                self.best_epoch = epoch
                print('【保存了best： acc提升】')
                return True

            elif round(self.best_score_by_acc / self.save_boundary, 2) == round(
                    val_acc / self.save_boundary, 2):
                if round(self.best_score_by_loss / self.save_boundary, 2) >= round(
                        val_loss / self.save_boundary, 2):
                    self.best_score_by_acc = val_acc
                    self.best_score_by_loss = val_loss
                    self.best_epoch = epoch
                    print('【保存了best：acc相同，loss降低】')
                    return True

        return False

    def get_now_best(self):
        return self.best_score_by_acc, self.best_score_by_loss, self.best_epoch

    def print_best_now(self):
        if self.best_score_by_acc == 0:
            print('未能满足best_score的条件')
        else:
            print(
                '当前【best】:acc:%.4f, loss:%.4f, epoch:%d, f1_score:%.4f' % (
                self.best_score_by_acc, self.best_score_by_loss, self.best_epoch + 1, self.best_f1_score))

    def print_best_final(self):
        print('best_score_by_acc :%.4f' % self.best_score_by_acc)
        print('best_score_by_loss :%.4f' % self.best_score_by_loss)
        print('best_score at epoch:%d' % (self.best_epoch + 1))
        print('best_f1_score:%.4f' % self.best_f1_score)

def get_nubclass_from_csv():
    readcsv = pd.read_csv(os.path.join(DATA_PATH, 'MedicalClass/train.csv'))
    file_unique = readcsv['label'].unique()
    list_unique = file_unique.tolist()
    return list_unique


if __name__ == '__main__':
    num_classes = len(get_nubclass_from_csv())
    dataset_wangyi = DatasetByWangyi(get_nubclass_from_csv())
    dataset_wangyi.set_Batch_Size([1]*209,[1]*209)
    train =dataset_wangyi.get_Next_Batch()
    print(train)