#!/usr/bin/env python
# coding:utf-8
"""
Name : WangyiUtilOnFlyai.py
Author  : 莫须有的嚣张
Contect : 291255700
Time    : 2019/7/28 上午9:42
Desc:
"""
from time import clock

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from flyai.source.base import DATA_PATH


from data_helper import *
from hyperparameter import random_per_epoch, reduce_lr_per_epochs, num_classes, train_vs_val
from path import DATA_PATH

DATA_ID = 'MedicalClass'

# 保留的图片中心的比例，如80%，写0.8，同时在prediction.py里要做修改
center_scale = 0.85


def get_1_x_data(image_path):
    img = Image.open(os.path.join(DATA_PATH, DATA_ID, image_path))

    return img


def get_1_y_data(label):

    return label

'''



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
'''

class dynamicBatchSize():
    def __init__(self, num_class):
        self.n = num_class
        self.now_batch = [1] * num_class

    def getSizebyAcc(self, acc, wrong_acc=None):
        normal_batch = [
            7 ,  # 0
            160 	,  # 1
            173 	,  # 2
            117 	,  # 3
            142 	,  # 4
            3 	,  # 5
            87 	,  # 6
            86 	,  # 7
            62 	,  # 8
            10 	,  # 9
            135 	,  # 10
            6 	,  # 11
            747 	,  # 12
            133 	,  # 13
            293 	,  # 14
            11 	,  # 15
            36 	,  # 16
            84 	,  # 17
            40 	,  # 18
            1 	,  # 19
            128 	,  # 20
            133 	,  # 21
            86 	,  # 22
            115 	,  # 23
            17 	,  # 24
            114 	,  # 25
            5 	,  # 26
            22 	,  # 27
            4 	,  # 28
            149 	,  # 29
            200 	,  # 30
            109 	,  # 31
            79 	,  # 32
            4 	,  # 33
            6 	,  # 34
            2 	,  # 35
            16 	,  # 36
            147 	,  # 37
            18 	,  # 38
            7 	,  # 39
            10 	,  # 40
            58 	,  # 41
            20 	,  # 42
            144 	,  # 43
            47 	,  # 44
            40 	,  # 45
            2 	,  # 46
            69 	,  # 47
            11 	,  # 48
            11 	,  # 49
            34 	,  # 50
            20 	,  # 51
            5 	,  # 52
            14 	,  # 53
            33 	,  # 54
            36 	,  # 55
            63 	,  # 56
            31 	,  # 57
            12 	,  # 58
            17 	,  # 59
            3 	,  # 60
            29 	,  # 61
            55 	,  # 62
            23 	,  # 63
            55 	,  # 64
            3 	,  # 65
            26 	,  # 66
            5 	,  # 67
            16 	,  # 68
            9 	,  # 69
            4 	,  # 70
            34 	,  # 71
            3 	,  # 72
            2 	,  # 73
            3 	,  # 74
            15 	,  # 75
            1 	,  # 76
            3 	,  # 77
            6 	,  # 78
            3 	,  # 79
            4 	,  # 80
            11 	,  # 81
            14 	,  # 82
            6 	,  # 83
            3 	,  # 84
            5 	,  # 85
            4 	,  # 86
            2 	,  # 87
            2 	,  # 88
            9 	,  # 89
            1 	,  # 90
            7 	,  # 91
            2 	,  # 92
            1 	,  # 93
            2 	,  # 94
            2 	,  # 95
            9 	,  # 96
            5 	,  # 97
            1 	,  # 98
            2 	,  # 99
            4 	,  # 100
            7 	,  # 101
            3 	,  # 102
            5 	,  # 103
            8 	,  # 104
            2 	,  # 105
            6 	,  # 106
            4 	,  # 107
            9 	,  # 108
            1 	,  # 109
            2 	,  # 110
            1 	,  # 111
            1 	,  # 112
            1 	,  # 113
            1 	,  # 114
            2 	,  # 115
            4 	,  # 116
            1 	,  # 117
            3 	,  # 118
            6 	,  # 119
            1 	,  # 120
            1 	,  # 121
            1 	,  # 122
            1 	,  # 123
            1 	,  # 124
            1 	,  # 125
            1 	,  # 126
            4 	,  # 127
            5 	,  # 128
            1 	,  # 129
            3 	,  # 130
            4 	,  # 131
            2 	,  # 132
            1 	,  # 133
            2 	,  # 134
            4 	,  # 135
            2 	,  # 136
            2 	,  # 137
            1 	,  # 138
            3 	,  # 139
            2 	,  # 140
            1 	,  # 141
            2 	,  # 142
            2 	,  # 143
            1 	,  # 144
            4 	,  # 145
            1 	,  # 146
            1 	,  # 147
            1 	,  # 148
            1 	,  # 149
            1 	,  # 150
            3 	,  # 151
            3 	,  # 152
            1 	,  # 153
            1 	,  # 154
            1 	,  # 155
            1 	,  # 156
            1 	,  # 157
            1 	,  # 158
            3 	,  # 159
            2 	,  # 160
            1 	,  # 161
            2 	,  # 162
            1 	,  # 163
            1 	,  # 164
            1 	,  # 165
            1 	,  # 166
            1 	,  # 167
            1 	,  # 168
            1 	,  # 169
            1 	,  # 170
            1 	,  # 171
            1 	,  # 172
            3 	,  # 173
            1 	,  # 174
            1 	,  # 175
            1 	,  # 176
            1 	,  # 177
            1 	,  # 178
            1 	,  # 179
            2 	,  # 180
            1 	,  # 181
            1 	,  # 182
            1 	,  # 183
            1 	,  # 184
            1 	,  # 185
            1 	,  # 186
            1 	,  # 187
            1 	,  # 188
            1 	,  # 189
            1 	,  # 190
            2 	,  # 191
            1 	,  # 192
            1 	,  # 193
            1 	,  # 194
            1 	,  # 195
            1 	,  # 196
            1 	,  # 197
            1 	,  # 198
            1 	,  # 199
            1 	,  # 200
            1 	,  # 201
            1 	,  # 202
            1 	,  # 203
            1 	,  # 204
            1 	,  # 205
            1 	,  # 206
            1 	,  # 207
            1 	,  # 208
            1 	,  # 209
            1 	,  # 210
            1 	,  # 211
            1 	,  # 212
            1 	,  # 213
            1 	,  # 214
            1 	,  # 215
            1 	,  # 216
            1 	,  # 217
            1 	,  # 218
            1 	,  # 219
            1 	,  # 220
            1 	,  # 221
            1 	,  # 222
            1 	,  # 223
            1 	,  # 224
            1 	,  # 225
            1 	,  # 226
            1 	,  # 227
            1 	,  # 228
            1 	,  # 229
            1 	,  # 230
            1 	,  # 231
            1 	,  # 232
            1 	,  # 233
            1 	,  # 234
            1 	,  # 235
            1 	,  # 236
            1 	,  # 237
            1 	,  # 238
            1 	,  # 239

        ]
        assert 0 <= acc <= 1
        self.now_batch = normal_batch
        # if acc < 0.75:
        #     # self.now_batch = [32] * self.n
        #     self.now_batch = normal_batch
        # elif acc < 0.8:
        #     self.now_batch = [3] * self.n
        # elif acc < 0.85:
        #     # self.now_batch = super_batch
        #     self.now_batch = [2] * self.n
        #
        # elif acc <= 1:
        #      self.now_batch= [1] * self.n

        if wrong_acc is not None and acc >= 0.75:
            wrong_acc_dict = {}
            for i, value in enumerate(wrong_acc):
                #  (value * 221) / normal_batch[1] 各类本身的错误率per wrong count/ per class
                self.now_batch[i] = int(self.now_batch[i] * (value * 240 / normal_batch[i]) + 1)
                wrong_acc_dict[i] = value

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
        val = self.get_Next_Val_Batch_fromMemory()
        return train ,val

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
        value_list = np.array(value_list)
        df = pd.DataFrame(value_list)
        df.columns = ['label', 'title', 'text']
        return df

    def get_Next_Val_Batch_fromMemory(self):
        value_list = []
        for iters in range(len(self.num_classes)):
            for i in range(self.val_batch_size[iters]):
                # 读取数据
                if len(self.val_pd_list[iters]['label'])==0:
                    continue
                value = self.val_pd_list[iters].values[self.val_step_list[iters]]
                value_list.append(value)
                # 修改游标
                self.val_step_list[iters] += 1
                self.val_step_list[iters] = self.val_step_list[iters] % self.val_total_list[iters]
        value_list = np.array(value_list)
        df = pd.DataFrame(value_list)
        df.columns = ['label', 'title', 'text']
        return df

def label_smoothing(inputs, epsilon=0.1):
    return (1.0 - epsilon) * inputs + epsilon / num_classes


class drawMatplotlib():
    def __init__(self):
        self.path = None
        self.train_acc_list = []  # x轴，对应每个epoch
        self.val_acc_list = []  # x轴，对应每个epoch
        self.f1_score = []  # x轴，对应每个epoch
        self.train_loss_list = []  # 对应每个epoch
        self.train_loss_list_y = []  # 每次batch训练的train acc
        self.val_loss_list = []  # 对应每个epoch
        self.learn_rate_list_y = []  # 收集用来matplotlib
        self.step_train_x = []  # 收集用来matplotlib
        self.step_val_x = []
        self.epoch_train_x = [] #acc,loss 的y轴坐标
        self.train_length_list = [] #acc,loss 的y轴坐标

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
        plt.plot(self.epoch_train_x, self.train_loss_list, label='train_loss')
        plt.plot(self.epoch_train_x, self.val_loss_list, label='val_loss')
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

        if val_acc > 0.30:
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
    train ,val =dataset_wangyi.get_Next_Batch()
    print(train)