## build CNN

from flyai.utils import remote_helper
from keras import Input, Model, losses
import keras
from keras.layers import *
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
import numpy as np

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

class Net():

    def __init__(self, num_classes ,label2id , text2id):
        """Declare all needed layers."""
        self.num_classes = num_classes
        self.net_choice = 'keras-bert'

        if self.net_choice == '构建cnn':
            rnn_unit_1 = 128  # RNN层包含cell个数
            embed_dim = 64  # 嵌入层大小
            class_num = len(label2id)
            num_word = len(text2id)
            MAX_SQUES_LEN = 68  # 最大句长

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
            self.model_cnn = keras.Model(text_input, pred)

        if self.net_choice == 'keras-bert':

            bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

            class_num = len(label2id)
            for l in bert_model.layers:
                l.trainable = True

            x1_in = Input(shape=(None,))
            x2_in = Input(shape=(None,))

            x = bert_model([x1_in, x2_in])
            x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
            p = Dense(class_num, activation='softmax')(x)

            self.model_cnn = Model([x1_in, x2_in], p)

    def get_Model(self):
        return self.model_cnn

    def cleanMemory(self):
        # if psutil.virtual_memory().percent > 90:
        #     print('内存占用率：', psutil.virtual_memory().percent, '现在启动model_cnn重置')
        #     tmp_model_path = os.path.join(os.curdir, 'data', 'output', 'model', 'reset_model_tmp.h5')
        #     self.model_cnn.save(tmp_model_path)  # creates a HDF5 file 'my_model.h5'
        #     del self.model_cnn  # deletes the existing model
        #     self.model_cnn = load_model(tmp_model_path)
        #     print('已重置了del model_cnn，防止内存泄露')
        # elif psutil.virtual_memory().percent > 80:
        #     print('内存占用率：', psutil.virtual_memory().percent, '%，将在90%重置model_cnn')
        pass
    def freezeConv(self):

        # self.model_cnn.get_layer(name='xception').trainable = False
        # # self.model_cnn.compile()
        pass

if __name__=='__main__':
    a=Net([1]*209,[1]*209,[1]*209).get_Model()
    print(a)
    a.summary()