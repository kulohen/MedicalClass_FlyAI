from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs

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
