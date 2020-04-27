import argparse

'''


'''

num_classes = 240

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

val_batch_size = [
7 	,	#	0
160 	,	#	1
173 	,	#	2
117 	,	#	3
142 	,	#	4
3 	,	#	5
87 	,	#	6
86 	,	#	7
62 	,	#	8
10 	,	#	9
135 	,	#	10
6 	,	#	11
747 	,	#	12
133 	,	#	13
293 	,	#	14
11 	,	#	15
36 	,	#	16
84 	,	#	17
40 	,	#	18
1 	,	#	19
128 	,	#	20
133 	,	#	21
86 	,	#	22
115 	,	#	23
17 	,	#	24
114 	,	#	25
5 	,	#	26
22 	,	#	27
4 	,	#	28
149 	,	#	29
200 	,	#	30
109 	,	#	31
79 	,	#	32
4 	,	#	33
6 	,	#	34
2 	,	#	35
16 	,	#	36
147 	,	#	37
18 	,	#	38
7 	,	#	39
10 	,	#	40
58 	,	#	41
20 	,	#	42
144 	,	#	43
47 	,	#	44
40 	,	#	45
2 	,	#	46
69 	,	#	47
11 	,	#	48
11 	,	#	49
34 	,	#	50
20 	,	#	51
5 	,	#	52
14 	,	#	53
33 	,	#	54
36 	,	#	55
63 	,	#	56
31 	,	#	57
12 	,	#	58
17 	,	#	59
3 	,	#	60
29 	,	#	61
55 	,	#	62
23 	,	#	63
55 	,	#	64
3 	,	#	65
26 	,	#	66
5 	,	#	67
16 	,	#	68
9 	,	#	69
4 	,	#	70
34 	,	#	71
3 	,	#	72
2 	,	#	73
3 	,	#	74
15 	,	#	75
1 	,	#	76
3 	,	#	77
6 	,	#	78
3 	,	#	79
4 	,	#	80
11 	,	#	81
14 	,	#	82
6 	,	#	83
3 	,	#	84
5 	,	#	85
4 	,	#	86
2 	,	#	87
2 	,	#	88
9 	,	#	89
0 	,	#	90
7 	,	#	91
2 	,	#	92
1 	,	#	93
2 	,	#	94
2 	,	#	95
9 	,	#	96
5 	,	#	97
0 	,	#	98
2 	,	#	99
4 	,	#	100
7 	,	#	101
3 	,	#	102
5 	,	#	103
8 	,	#	104
2 	,	#	105
6 	,	#	106
4 	,	#	107
9 	,	#	108
1 	,	#	109
2 	,	#	110
1 	,	#	111
1 	,	#	112
0 	,	#	113
1 	,	#	114
2 	,	#	115
4 	,	#	116
1 	,	#	117
3 	,	#	118
6 	,	#	119
1 	,	#	120
1 	,	#	121
1 	,	#	122
0 	,	#	123
1 	,	#	124
1 	,	#	125
1 	,	#	126
4 	,	#	127
5 	,	#	128
0 	,	#	129
3 	,	#	130
4 	,	#	131
2 	,	#	132
0 	,	#	133
2 	,	#	134
4 	,	#	135
2 	,	#	136
2 	,	#	137
1 	,	#	138
3 	,	#	139
2 	,	#	140
1 	,	#	141
2 	,	#	142
2 	,	#	143
1 	,	#	144
4 	,	#	145
1 	,	#	146
1 	,	#	147
0 	,	#	148
1 	,	#	149
1 	,	#	150
3 	,	#	151
3 	,	#	152
1 	,	#	153
1 	,	#	154
0 	,	#	155
1 	,	#	156
0 	,	#	157
0 	,	#	158
3 	,	#	159
2 	,	#	160
1 	,	#	161
2 	,	#	162
1 	,	#	163
0 	,	#	164
0 	,	#	165
0 	,	#	166
0 	,	#	167
1 	,	#	168
0 	,	#	169
0 	,	#	170
1 	,	#	171
0 	,	#	172
3 	,	#	173
0 	,	#	174
1 	,	#	175
1 	,	#	176
1 	,	#	177
0 	,	#	178
0 	,	#	179
2 	,	#	180
0 	,	#	181
1 	,	#	182
0 	,	#	183
0 	,	#	184
1 	,	#	185
0 	,	#	186
0 	,	#	187
1 	,	#	188
0 	,	#	189
0 	,	#	190
2 	,	#	191
0 	,	#	192
0 	,	#	193
1 	,	#	194
0 	,	#	195
0 	,	#	196
0 	,	#	197
0 	,	#	198
0 	,	#	199
0 	,	#	200
1 	,	#	201
1 	,	#	202
0 	,	#	203
0 	,	#	204
0 	,	#	205
0 	,	#	206
0 	,	#	207
0 	,	#	208
0 	,	#	209
0 	,	#	210
1 	,	#	211
0 	,	#	212
0 	,	#	213
0 	,	#	214
0 	,	#	215
0 	,	#	216
0 	,	#	217
0 	,	#	218
0 	,	#	219
0 	,	#	220
0 	,	#	221
0 	,	#	222
0 	,	#	223
0 	,	#	224
0 	,	#	225
0 	,	#	226
0 	,	#	227
0 	,	#	228
0 	,	#	229
0 	,	#	230
0 	,	#	231
0 	,	#	232
0 	,	#	233
0 	,	#	234
0 	,	#	235
0 	,	#	236
0 	,	#	237
0 	,	#	238
0 	,	#	239
]

[
136	,	#	0
2934	,	#	1
3173	,	#	2
2147	,	#	3
2610	,	#	4
48	,	#	5
1603	,	#	6
1582	,	#	7
1132	,	#	8
180	,	#	9
2471	,	#	10
108	,	#	11
13685	,	#	12
2435	,	#	13
5359	,	#	14
209	,	#	15
657	,	#	16
1533	,	#	17
729	,	#	18
13	,	#	19
2347	,	#	20
2439	,	#	21
1583	,	#	22
2098	,	#	23
312	,	#	24
2097	,	#	25
93	,	#	26
403	,	#	27
70	,	#	28
2739	,	#	29
3663	,	#	30
1998	,	#	31
1447	,	#	32
82	,	#	33
118	,	#	34
37	,	#	35
290	,	#	36
2694	,	#	37
335	,	#	38
122	,	#	39
188	,	#	40
1061	,	#	41
365	,	#	42
2640	,	#	43
853	,	#	44
734	,	#	45
43	,	#	46
1273	,	#	47
205	,	#	48
206	,	#	49
631	,	#	50
367	,	#	51
100	,	#	52
250	,	#	53
597	,	#	54
663	,	#	55
1151	,	#	56
562	,	#	57
217	,	#	58
305	,	#	59
53	,	#	60
526	,	#	61
1014	,	#	62
416	,	#	63
1010	,	#	64
63	,	#	65
468	,	#	66
97	,	#	67
296	,	#	68
159	,	#	69
76	,	#	70
617	,	#	71
48	,	#	72
31	,	#	73
58	,	#	74
280	,	#	75
20	,	#	76
60	,	#	77
104	,	#	78
50	,	#	79
75	,	#	80
195	,	#	81
253	,	#	82
109	,	#	83
53	,	#	84
95	,	#	85
69	,	#	86
31	,	#	87
36	,	#	88
161	,	#	89
7	,	#	90
122	,	#	91
38	,	#	92
18	,	#	93
43	,	#	94
30	,	#	95
166	,	#	96
101	,	#	97
3	,	#	98
38	,	#	99
65	,	#	100
132	,	#	101
48	,	#	102
85	,	#	103
147	,	#	104
41	,	#	105
112	,	#	106
71	,	#	107
156	,	#	108
12	,	#	109
29	,	#	110
27	,	#	111
15	,	#	112
8	,	#	113
23	,	#	114
37	,	#	115
75	,	#	116
15	,	#	117
54	,	#	118
109	,	#	119
24	,	#	120
17	,	#	121
28	,	#	122
6	,	#	123
24	,	#	124
16	,	#	125
13	,	#	126
76	,	#	127
94	,	#	128
8	,	#	129
58	,	#	130
73	,	#	131
38	,	#	132
9	,	#	133
38	,	#	134
72	,	#	135
32	,	#	136
46	,	#	137
20	,	#	138
64	,	#	139
45	,	#	140
21	,	#	141
39	,	#	142
42	,	#	143
18	,	#	144
70	,	#	145
13	,	#	146
14	,	#	147
2	,	#	148
12	,	#	149
13	,	#	150
47	,	#	151
48	,	#	152
18	,	#	153
12	,	#	154
2	,	#	155
14	,	#	156
8	,	#	157
4	,	#	158
61	,	#	159
30	,	#	160
13	,	#	161
33	,	#	162
14	,	#	163
5	,	#	164
2	,	#	165
5	,	#	166
7	,	#	167
11	,	#	168
9	,	#	169
8	,	#	170
19	,	#	171
4	,	#	172
55	,	#	173
2	,	#	174
15	,	#	175
14	,	#	176
21	,	#	177
8	,	#	178
4	,	#	179
29	,	#	180
7	,	#	181
24	,	#	182
2	,	#	183
8	,	#	184
24	,	#	185
4	,	#	186
9	,	#	187
11	,	#	188
6	,	#	189
5	,	#	190
32	,	#	191
8	,	#	192
3	,	#	193
10	,	#	194
3	,	#	195
4	,	#	196
4	,	#	197
2	,	#	198
5	,	#	199
7	,	#	200
14	,	#	201
16	,	#	202
5	,	#	203
3	,	#	204
3	,	#	205
3	,	#	206
6	,	#	207
2	,	#	208
1	,	#	209
2	,	#	210
12	,	#	211
5	,	#	212
5	,	#	213
4	,	#	214
3	,	#	215
2	,	#	216
1	,	#	217
5	,	#	218
1	,	#	219
1	,	#	220
3	,	#	221
5	,	#	222
2	,	#	223
3	,	#	224
1	,	#	225
3	,	#	226
2	,	#	227
1	,	#	228
1	,	#	229
3	,	#	230
1	,	#	231
5	,	#	232
1	,	#	233
1	,	#	234
2	,	#	235
1	,	#	236
1	,	#	237
1	,	#	238
1	,	#	239


]

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