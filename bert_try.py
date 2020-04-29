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

# from data_helper import *
from WangyiUtilOnFlyai import *
from net import *

if __name__ == '__main__':

    '''
    分割线
    '''

    dataset_wangyi = DatasetByWangyi(get_nubclass_from_csv())
    dataset_wangyi.set_Batch_Size([1]*209,[1]*209)

    text2id, _ = load_dict(os.path.join(DATA_PATH, 'MedicalClass/words_fr.dict'))
    label2id, _ = load_labeldict(os.path.join(DATA_PATH, 'MedicalClass/label.dict'))

    model_net = Net(num_classes=get_nubclass_from_csv(), label2id=label2id, text2id=text2id)
    model = model_net.get_Model()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    for i in range(100):
        train_batch, val_batch = dataset_wangyi.get_Next_Batch()
        train_text_x1, train_text_x2,train_label = read_data_v2(train_batch, text2id, label2id)

        # predicts = model.predict([train_text_x1, train_text_x2])
        # print('over')
        model.fit( [train_text_x1, train_text_x2], np.array(train_label),
                                      # validation_data=(np.array(x_val), np.array(y_val)),
                                      batch_size=16, verbose=1)