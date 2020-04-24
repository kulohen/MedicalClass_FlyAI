## build CNN
import os

import psutil
from flyai.utils import remote_helper

from hyperparameter import img_size
import torch
from torch import nn
# from senet.se_resnet import se_resnet50
from  torchvision.models.resnet import ResNet,resnet50,resnext101_32x8d
from torchvision.models.densenet import densenet201,densenet161
from  torchvision.models.inception import *
# from  torchsummary import summary
from efficientnet_pytorch import EfficientNet
import keras
from keras.layers import *

class Net():

    def __init__(self, num_classes ,label2id , text2id):
        """Declare all needed layers."""
        self.num_classes = num_classes
        self.net_choice = '构建cnn'

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

        if self.net_choice =='resnet50':
            # resnet50
            weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/resnet50-19c8e357.pth')
            pre = torch.load(weights_path)
            # self.model_cnn = resnet50(num_classes=num_classes, pretrained=True)
            self.model_cnn = resnet50(pretrained=True)
            self.model_cnn.load_state_dict(pre)
            feature = self.model_cnn.fc.in_features
            self.model_cnn.fc = torch.nn.Linear(in_features=feature, out_features=num_classes)

        if self.net_choice == 'densenet201':
            # densenet201
            weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/densenet201-c1103571.pth')
            pre = torch.load(weights_path)
            self.model_cnn = densenet201(pretrained=True)
            self.model_cnn.load_state_dict(pre)
            feature = self.model_cnn._fc.in_features
            self.model_cnn._fc = torch.nn.Linear(in_features=feature, out_features=num_classes)

        if self.net_choice == 'densenet161':
            # densenet161
            weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/densenet161-8d451a50.pth')
            pre = torch.load(weights_path)
            self.model_cnn = densenet161(pretrained=True)
            self.model_cnn.load_state_dict(pre,False)
            feature = self.model_cnn.classifier.in_features
            self.model_cnn.classifier = torch.nn.Linear(in_features=feature, out_features=num_classes)

        if self.net_choice == 'resnext101_32x8d':
            # resnext101_32x8d
            weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/resnext101_32x8d-8ba56ff5.pth')
            pre = torch.load(weights_path)
            self.model_cnn = resnext101_32x8d(pretrained=True)
            self.model_cnn.load_state_dict(pre)
            self.model_cnn.fc = torch.nn.Linear(2048, num_classes)

        if self.net_choice == 'efficientnet_b3':
            # efficientnet_b3
            weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b3-5fb5a3c3.pth')
            pre = torch.load(weights_path)
            # self.model_cnn = EfficientNet.from_pretrained('efficientnet-b3')
            self.model_cnn = EfficientNet.from_name('efficientnet-b3')
            self.model_cnn.load_state_dict(pre)
            feature = self.model_cnn._fc.in_features
            self.model_cnn._fc = torch.nn.Linear(in_features=feature, out_features=num_classes)
            self.model_cnn._dropout = torch.nn.Dropout(p=0.4, inplace=False)

        if self.net_choice == 'efficientnet_b5':
            # efficientnet_b5
            weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b5-b6417697.pth')
            pre = torch.load(weights_path)
            self.model_cnn = EfficientNet.from_name('efficientnet-b5')
            self.model_cnn.load_state_dict(pre)
            feature = self.model_cnn._fc.in_features
            self.model_cnn._fc = torch.nn.Linear(in_features=feature, out_features=num_classes)

        if self.net_choice == 'efficientnet_b7':
            # efficientnet_b7
            weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b7-dcc49843.pth')
            pre = torch.load(weights_path)
            self.model_cnn = EfficientNet.from_name('efficientnet-b7')
            self.model_cnn.load_state_dict(pre)
            feature = self.model_cnn._fc.in_features
            self.model_cnn._fc = torch.nn.Linear(in_features=feature, out_features=num_classes)

        if self.net_choice == 'efficientnet_b1':
            # efficientnet_b1
            weights_path = remote_helper.get_remote_date('https://www.flyai.com/m/efficientnet-b1-f1951068.pth')
            pre = torch.load(weights_path)
            self.model_cnn = EfficientNet.from_name('efficientnet-b1')
            self.model_cnn.load_state_dict(pre)
            feature = self.model_cnn._fc.in_features
            self.model_cnn._fc = torch.nn.Linear(in_features=feature, out_features=num_classes)
            self.model_cnn._dropout = torch.nn.Dropout(p=0.3, inplace=False)

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