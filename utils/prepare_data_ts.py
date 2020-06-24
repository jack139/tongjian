# -*- coding: utf-8 -*-

import re
import codecs
import pandas as pd
import numpy as np
from sklearn import preprocessing
import jiebazhc as jieba
from . import utils

X_BLANK = 'blank'
y_BLANK = '让我想想'

# 诗句分词
def pre_load(filename):
    content = utils.load_content_lines(filename)

    f = codecs.open(filename+'.2.txt', 'w', 'utf-8')

    for l in content:
        l = l.strip()
        if len(l)==0 or l[0]=='#': # 标题，忽略
            continue

        line = re.split('，|。|？|！', l)
        line = [i for i in line if len(i)>0]
        if len(line)!=2:
            print(line)
        #if len(line[0])!=len(line[1]):
        #    print(line)

        
        left = jieba.cut(line[0])
        right = jieba.cut(line[1])

        f.write('%s -- %s\n'% (' '.join([i for i in left]), ' '.join([i for i in right])) )

    f.close()


# 分词后诗句处理 （一般需手工核对后）
def load_data(filename):
    content = utils.load_content(filename)

    content1 = content.replace('--','')
    segs = content1.split()
    segments = [i for i in segs if len(i)>0]
    segmentDF = pd.DataFrame({'segment':segments})
    #segmentDF = utils.remove_stopwords(segmentDF)

    #对词频进行统计。

    segStat = segmentDF.groupby(by=["segment"])["segment"].agg(count="count").reset_index().sort_values(by=['count'],ascending=False)
    #print(segStat.head(100))

    # 准备 X 

    # 词标签规范化
    leX = preprocessing.LabelEncoder()
    leX.fit(list(segStat['segment'])+[X_BLANK, y_BLANK])

    # 取得空位的label
    blank_label = leX.transform([X_BLANK])[0]
    blank_label_y = leX.transform([y_BLANK])[0]


    # 标签化
    lines = content.split('\n')
    lines = [i for i in lines if len(i)>0]

    lines_data = []

    # 词标签化
    # 词特征组成： 词本身标签 + 前一个词标签 + 后一个词标签 ， 如果前、后有空，则空标签
    X_data = []
    y_data = [] 
    for line in lines:
        if len(line)==0:
            continue
        left, right = line.split('--')
        left = left.split()
        right = right.split()

        # 检查两句分词个数是否一致
        if len(left)!=len(right):
            print('skip:', line)
            continue

        # 检查两句分词，每词字数是否一致
        x=[len(left[i])==len(right[i]) for i in range(len(left))]
        if False in x:
            print('skip:', line)
            continue

        right_df = pd.DataFrame({'segment':right})
        right_data = leX.transform(right_df['segment'])

        for i in range(len(left)):
            # 每个词生成特征

            # 前一个词
            if i==0:
                last_word = X_BLANK
            else:
                last_word = left[i-1]

            # 后一个词
            if i+1==len(left):
                next_word = X_BLANK
            else:
                next_word = left[i+1]

            # 词特征标签
            word_feature = [left[i], last_word, next_word]
            word_df = pd.DataFrame({'segment':word_feature})    
            word_data = leX.transform(word_df['segment'])

            X_data.append(word_data)
            y_data.append(right_data[i])

            # 词特征标签2, 不考虑位置
            word_feature = [left[i], X_BLANK, X_BLANK]
            word_df = pd.DataFrame({'segment':word_feature})    
            word_data = leX.transform(word_df['segment'])

            X_data.append(word_data)
            y_data.append(right_data[i])


    # 添加空词标签
    word_feature = [X_BLANK, X_BLANK, X_BLANK]
    word_df = pd.DataFrame({'segment':word_feature})    
    word_data = leX.transform(word_df['segment'])
    X_data.append(word_data)
    y_data.append(blank_label_y)


    # 每扩充为等长
    max_len = max(map(len, X_data))
    #print(max_len)
    X_data = [ np.append(l[:max_len], [blank_label]*(max_len-len(l))) for l in X_data ]

    return X_data, y_data, leX, leX, max_len
