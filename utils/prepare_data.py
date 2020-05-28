# -*- coding: utf-8 -*-

import re
import jiebazhc as jieba
import pandas as pd
import numpy as np
from sklearn import preprocessing
from . import utils

X_BLANK = 'blank'
y_BLANK = '让我想想'

def load_data(filename):
    content = utils.load_content(filename)

    segs = jieba.cut(content)
    segments = [i for i in segs if len(i)>0]
    segmentDF = pd.DataFrame({'segment':segments})
    segmentDF = utils.remove_stopwords(segmentDF)

    #对词频进行统计。

    segStat = segmentDF.groupby(by=["segment"])["segment"].agg(count="count").reset_index().sort_values(by=['count'],ascending=False)
    #print(segStat.head(100))

    # 准备 X 

    # 词标签规范化
    leX = preprocessing.LabelEncoder()
    leX.fit(list(segStat['segment'])+[X_BLANK])

    # 取得空位的label
    blank_label = leX.transform([X_BLANK])[0]

    #list(le.classes_)
    #le.transform(["tokyo", "tokyo", "paris"])
    #list(le.inverse_transform([2, 2, 1]))

    # 标签化
    lines = content.replace('：','').replace('\n','').replace('\r','')
    lines = re.split('。|？|“|”|‘|’|！|；', lines)
    lines = [i for i in lines if len(i)>0]

    lines_data = []

    # 词标签化
    X_data = []
    y_lines = [] 
    for line in lines:
        if len(line)==0:
            continue
        line_segs = jieba.cut(line)
        line_segs = [i for i in line_segs if len(i)>0]
        if len(line_segs)==0:
            continue
        line_df = pd.DataFrame({'segment':line_segs})
        line_df = utils.remove_stopwords(line_df)
        line_data = leX.transform(line_df['segment'])
        X_data.append(line_data)
        y_lines.append(line)

    # 每扩充为等长
    max_len = max(map(len, X_data))
    #print(max_len)
    X_data = [ np.append(l[:max_len], [blank_label]*(max_len-len(l))) for l in X_data ]

    # 准备y

    #句子标签规范化
    ley = preprocessing.LabelEncoder()
    ley.fit(y_lines+[y_BLANK])
    y_data = ley.transform(y_lines).tolist()

    # 取得y空位的label
    blank_label_y = ley.transform([y_BLANK])[0]

    # X 末尾增一行 blank；y 头删一行， 末尾增两行 blank， 错行对应， 上句对下句
    X_data.append(np.array([blank_label]*max_len))
    y_data = y_data[1:]
    y_data.append(blank_label_y)
    y_data.append(blank_label_y)

    return X_data, y_data, leX, ley, max_len
