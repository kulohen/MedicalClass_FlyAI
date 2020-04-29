'''
https://github.com/CyberZHG/keras-bert/blob/master/demo/load_model/load_and_extract.py
'''

import sys
import numpy as np
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
from flyai.utils import remote_helper

print('This demo demonstrates how to load the pre-trained model and extract word embeddings')
path = remote_helper.get_remote_date('https://www.flyai.com/m/chinese_L-12_H-768_A-12.zip')
config_path = 'data/input/model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'data/input/model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'data/input/model/chinese_L-12_H-768_A-12/vocab.txt'

if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    from keras_bert.datasets import get_pretrained, PretrainedList
    # model_path = get_pretrained(PretrainedList.chinese_base)
    # model_path = get_pretrained(path)

# paths = get_checkpoint_paths(model_path)

# model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=10)
model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=10)
model.summary(line_length=120)

token_dict = load_vocabulary(dict_path)

tokenizer = Tokenizer(token_dict)
text = '语言模型是不是'
tokens = tokenizer.tokenize(text)
print('Tokens:', tokens)
indices, segments = tokenizer.encode(first=text, max_len=10)

predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])

"""Official outputs:
{
  "linex_index": 0,
  "features": [
    {
      "token": "[CLS]",
      "layers": [
        {
          "index": -1,
          "values": [-0.63251, 0.203023, 0.079366, -0.032843, 0.566809, ...]
        }
      ]
    },
    {
      "token": "语",
      "layers": [
        {
          "index": -1,
          "values": [-0.758835, 0.096518, 1.071875, 0.005038, 0.688799, ...]
        }
      ]
    },
    {
      "token": "言",
      "layers": [
        {
          "index": -1,
          "values": [0.547702, -0.792117, 0.444354, -0.711265, 1.20489, ...]
        }
      ]
    },
    {
      "token": "模",
      "layers": [
        {
          "index": -1,
          "values": [-0.292423, 0.605271, 0.499686, -0.42458, 0.428554, ...]
        }
      ]
    },
    {
      "token": "型",
      "layers": [
        {
          "index": -1,
          "values": [ -0.747346, 0.494315, 0.718516, -0.872353, 0.83496, ...]
        }
      ]
    },
    {
      "token": "[SEP]",
      "layers": [
        {
          "index": -1,
          "values": [-0.874138, -0.216504, 1.338839, -0.105871, 0.39609, ...]
        }
      ]
    }
  ]
}
"""