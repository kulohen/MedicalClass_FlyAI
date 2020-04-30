# -*- coding: utf-8 -*
import os
import numpy as np
from flyai.framework import FlyAI
from path import MODEL_PATH, DATA_PATH
from keras.models import load_model
from data_helper import pred_process, load_dict, load_labeldict
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
from flyai.utils import remote_helper
from keras_bert import get_custom_objects


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

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在构造方法中加载模型
        '''
        self.text2id, _ = load_dict(os.path.join(DATA_PATH, 'MedicalClass/words_fr.dict'))
        _, self.id2label = load_labeldict(os.path.join(DATA_PATH, 'MedicalClass/label.dict'))
        self.model = load_model(os.path.join(MODEL_PATH, 'model.h5') , custom_objects=get_custom_objects())

    def predict(self, title, text):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"title": "心率为72bpm是正常的吗", "text": "最近不知道怎么回事总是感觉心脏不舒服..."}
        :return: 模型预测成功中户 {"label": "心血管科"}
        '''
        text_line = title +text
        X1, X2 = [], []
        tokens = tokenizer.tokenize(text_line)

        x1, x2 = tokenizer.encode(first=text_line, max_len=68)
        X1.append(x1)
        X2.append(x2)
        # pred = self.id2label[np.argmax(self.model.predict(np.array(text_line)))]
        pred = self.id2label[np.argmax(self.model.predict([np.array(X1),np.array(X2)]))]
        return {'label': pred}

if __name__ == '__main__':
    p = Prediction()
    p.load_model()
    title = "心率为72bpm是正常的吗"
    text  = "最近不知道怎么回事总是感觉心脏不舒服..."
    result = p.predict(title,text)
    print(result)