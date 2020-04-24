import argparse

'''


'''

num_classes = 21

# 一、构建网络
# net.py修改


# 二、数据结构

img_size = [224,224] # 同时需要在prediction.py文件中修改，并没有关联

# 数据增强倍数
per_train_ratio = 1

# random-crop 裁剪的比例
scale_num = 0.9

# 保存最佳model的精准度，比如1%的准确范围写1,若2%的保存范围写2
save_boundary =0.1

train_vs_val = 0.85
# 三、成绩（调参）：

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=300, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=64, type=int, help="batch size")
args = parser.parse_args()



train_epoch = args.EPOCHS
train_batch = args.BATCH


# 随机学习率启动per epoch
random_per_epoch = 15

# 多岁epochs后降低学习率
reduce_lr_per_epochs = 6


# 四、性能or速度
#是否开启每一类的验证,True代表开启（影响速率）
val_per_class = False

# 训练集的每类的batch的量，组成的list
train_batch_List = [ 150 ] * num_classes
# 验证集的batch量，模拟预测集

val_batch_size = []

class_weights_per_21 = {
0	:	6.6969	,
1	:	2.5998	,
2	:	3.4671	,
3	:	3.9165	,
4	:	2.4906	,
5	:	0.42	,
6	:	0.1953	,
7	:	0.2919	,
8	:	0.1008	,
9	:	0.1617	,
10	:	0.1953	,
11	:	0.063	,
12	:	0.0231	,
13	:	0.1764	,
14	:	0.0378	,
15	:	0.0483	,
16	:	0.0147	,
17	:	0.0336	,
18	:	0.0105	,
19	:	0.042	,
20	:	0.0189
}



param_list =[
    '【参数情况】'
    '一/构建网络',
    '框架/神经网络修改  ',
    '冻结训练层  ',
    'learn transfer : %s'%'是',
    '激活函数linear line 非relu sigmoid',
    '',
    '二、数据结构',
    '训练数据平衡,%s'%'是',
    '图片分辨率,%d:%d'%(img_size[0],img_size[1]),
    '重置train:val的数据量比例,%.2f'%train_vs_val,
    '数据增强倍数,%.2f'%per_train_ratio,
    'random-crop 裁剪的比例：%.2f'%scale_num,
    '保存model的条件,%.2f'%save_boundary,
    'Train set dropout0.5（一定程度避开噪音，不一定奏效）  ,%s'%'是',
    '',
    '三、成绩（调参）：  ',
    'epoch 介于15-20  ，%d'%train_epoch,
    'learn rate  ',
    'batch影响梯度  ,%d'%train_batch,
    '',
    '四、性能or速度  ',
    '减少val_batch的量  ',
    '轻度框架  '
]


#  print参数，以便在flyai log里查看
def printHpperparameter():
    for s in param_list:
        print(s)
    pass


if __name__=='__main__':
    # printHpperparameter()
    # print(param_list_dict)
    print(val_batch_size)
    print((val_batch_size * [100]))