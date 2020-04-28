from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs

import codecs
import os
import sys

import numpy as np
from keras import Input, Model, losses
from keras.layers import Lambda, Dense
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.activations import softmax, sigmoid
from keras_bert import Tokenizer, load_trained_model_from_checkpoint

from data_helper import *
from WangyiUtilOnFlyai import *
# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper
path = remote_helper.get_remote_date('https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip')
config_path = 'data/input/model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'data/input/model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'data/input/model/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)
print(tokenizer.tokenize(u'今天天气不错'))
# 输出是 ['[CLS]', u'今', u'天', u'天', u'气', u'不', u'错', '[SEP]']

def build_bert_model(X1, X2):
    '''
    :param X1:经过编码过后的集合
    :param X2:经过编码过后的位置集合
    :return:模型
    '''

    #  ！！！！！！ 非常重要的！！！非常重要的！！！非常重要的！！！
    # 加载  Google 训练好的模型bert 就一句话，非常完美prefect
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    # config_path 是Bert模型的参数，checkpoint_path 是Bert模型的最新点，即训练的最新结果
    # 特别注意的是  加载Bert的路径 问题，
    # 注：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip，
    #     下载完之后，解压得到4个文件，直接放到 项目的路径下，要写上绝对路径，以防出现问题。
    # 安装 keras-bert：pip install keras-bert
    wordvec = bert_model.predict([X1, X2])
    # wordvec就是得到的向量矩阵
    return wordvec


def build_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(1, activation=sigmoid))
    model.compile(loss=losses.binary_crossentropy, optimizer=Adam(1e-5), metrics=['accuracy'])

    return model

def get_token_dict(dict_path):
    '''
    :param: dict_path: 是bert模型的vocab.txt文件
    :return:将文件中字进行编码
    '''
    # 将bert模型中的 字 进行编码
    # 目的是 喂入模型  的是  这些编码，不是汉字
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)


# 得到编码

def get_encode(pos, neg, token_dict):
    '''
    :param pos:第一类文本数据
    :param neg:第二类文本数据
    :param token_dict:编码字典
    :return:[X1,X2]，其中X1是经过编码后的集合，X2表示第一句和第二句的位置，记录的是位置信息
    '''

    all_data = pos + neg
    tokenizer = OurTokenizer(token_dict)
    X1 = []
    X2 = []
    for line in all_data:
        x1, x2 = tokenizer.encode(first=line)
    X1.append(x1)
    X2.append(x2)
    # 利用Keras API进行对数据集  补齐  操作。
    # 与word2vec没什么区别，都需要进行补齐
    X1 = sequence.pad_sequences(X1, maxlen=128, padding='post', truncating='post')
    X2 = sequence.pad_sequences(X2, maxlen=128, padding='post', truncating='post')
    return [X1, X2]


if __name__ == '__main__':
    # 注意，尽管可以设置seq_len=None，但是仍要保证序列长度不超过512
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()

    '''
    分割线
    '''

    # bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    # # bert_model.summary()
    # bert_model.compile(optimizer=Adam(lr=0.0003), loss='categorical_crossentropy', metrics=['acc'])
    # print(bert_model)
    #
    # num_classes = len(get_nubclass_from_csv())
    # dataset_wangyi = DatasetByWangyi(get_nubclass_from_csv())
    # dataset_wangyi.set_Batch_Size([1]*209,[1]*209)
    # train_batch ,val_batch =dataset_wangyi.get_Next_Batch()
    # # print(train)
    # text2id, _ = load_dict(os.path.join(DATA_PATH, 'MedicalClass/words_fr.dict'))
    # label2id, _ = load_labeldict(os.path.join(DATA_PATH, 'MedicalClass/label.dict'))
    # train_text, train_label = read_data(train_batch, text2id, label2id)
    # x_train, y_train = get_batches(train_text, train_label, text_padding=text2id['_pad_'])
    #
    #
    # bert_model.fit( np.array(x_train), np.array(y_train),
    #                               # validation_data=(np.array(x_val), np.array(y_val)),
    #                               batch_size=64, verbose=1)