# -*- coding: utf-8 -*- 
# author: Honay.King

import os
import json
import jieba
import numpy as np
from net import tokenizer

def load_dict(dictFile):
    if not os.path.exists(dictFile):
        print('[ERROR] load_dict failed! | The params {}'.format(dictFile))
        return None
    with open(dictFile, 'r', encoding='UTF-8') as df:
        dictF = json.load(df)
    text2id, id2text = dict(), dict()
    count = 0
    for key, value in dictF.items():
        text2id[key] = count
        id2text[count] = key
        count += 1
    return text2id, id2text


def load_labeldict(dictFile):
    if not os.path.exists(dictFile):
        print('[ERROR] load_labeldict failed! | The params {}'.format(dictFile))
        return None
    with open(dictFile, 'r', encoding='UTF-8') as df:
        label2id = json.load(df)
    id2label = dict()
    for key, value in label2id.items():
        id2label[value] = key
    return label2id, id2label

'''
jieba 的词向量
'''
def read_data(data, textdict, labeldict):
    text_data, label_data = list(), list()
    for ind, row in data.iterrows():
        # jieba.lcut("中国是一个伟大的国家")
        # 　['中国', '是', '一个', '伟大', '的', '国家']
        text_line = jieba.lcut(row['title'] + row['text'])
        tmp_text = list()
        for text in text_line:
            if text in textdict.keys():
                tmp_text.append(textdict[text])
            else:
                tmp_text.append(textdict['_unk_'])
        text_data.append(tmp_text)
        label = np.zeros(len(labeldict), dtype=int)
        label[labeldict[row['label']]] = 1
        label_data.append(label)
    return text_data, label_data

'''
keras-bert 版本
'''
def read_data_v2(data, textdict, labeldict):
    text_data, label_data = list(), list()
    X1, X2 = [], []
    for ind, row in data.iterrows():

        # text_line = jieba.lcut(row['title'] + row['text'])
        text_line = row['title'] + row['text']
        # X1,X2 = [],[]
        # for text in text_line:
        #     x1, x2 = tokenizer.encode(first=text, max_len=68)  # bert的token办法
        tokens = tokenizer.tokenize(text_line)

        x1,x2 = tokenizer.encode(first=text_line, max_len=30)
        # X1 = seq_padding(X1)
        # X2 = seq_padding(X2)
        X1.append(x1)
        X2.append(x2)

        label = np.zeros(len(labeldict), dtype=int)
        label[labeldict[row['label']]] = 1
        label_data.append(label)

    return np.array(X1), np.array(X2), label_data

'''
keras-bert的padding办法
'''
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def pred_process(title, text, textdict, max_len=68):
    text_line = jieba.lcut(title+text)
    tmp_text = list()
    for item in text_line:
        if item in textdict.keys():
            tmp_text.append(textdict[item])
        else:
            tmp_text.append(textdict['_unk_'])
        if len(tmp_text) >= max_len:
            tmp_text = tmp_text[:max_len]
        else:
            tmp_text = tmp_text + [textdict['_pad_']] * (max_len - len(tmp_text))
    return [np.array(tmp_text)]


def batch_padding(text_batch, padding, max_len=68):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的text_length
    参数：
    - text_batch
    - padding: <PAD>对应索引号
    '''
    batch_text = list()
    for text in text_batch:
        if len(text) >= max_len:
            batch_text.append(np.array(text[:max_len]))
        else:
            batch_text.append(np.array(text + [padding] * (max_len-len(text))))
    return batch_text

'''
def get_batches(texts, labels, batch_size, text_padding):
    for batch_i in range(0, len(labels) // batch_size):
        start_i = batch_i * batch_size
        texts_batch = texts[start_i: start_i + batch_size]
        labels_batch = labels[start_i: start_i + batch_size]

        pad_texts_batch = batch_padding(texts_batch, text_padding)
        yield pad_texts_batch, labels_batch
'''
def get_batches(texts, labels, text_padding):

    pad_texts_batch = batch_padding(texts, text_padding)
    return pad_texts_batch, labels

'''
def get_val_batch(texts, labels, batch_size, text_padding):
    texts_batch = texts[:batch_size]
    labels_batch = labels[:batch_size]
    pad_texts_batch = batch_padding(texts_batch, text_padding)
    return pad_texts_batch, labels_batch
'''



if __name__ == "__main__":
    from prediction import Prediction

    model = Prediction()
    model.load_model()

    result = model.predict(title='甲状腺功能减退能治好吗？', text='无')
    print(result)

    exit(0)