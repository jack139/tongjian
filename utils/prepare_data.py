# -*- coding: utf-8 -*-

import jiebazhc as jieba
import pandas as pd
import numpy as np
from sklearn import preprocessing
from . import utils

def load_data(filename):
    content = utils.load_content(filename)

    segs = jieba.cut(content)
    segments = [i for i in segs if len(i)>1]
    segmentDF = pd.DataFrame({'segment':segments})
    segmentDF = utils.remove_stopwords(segmentDF)

    #对词频进行统计。

    segStat = segmentDF.groupby(by=["segment"])["segment"].agg(count="count").reset_index().sort_values(by=['count'],ascending=False)
    #print(segStat.head(100))

    # 词标签规范化
    leX = preprocessing.LabelEncoder()
    leX.fit(list(segStat['segment'])+['<blank>'])

    # 取得空位的label
    blank_label = leX.transform(['<blank>'])

    #list(le.classes_)
    #le.transform(["tokyo", "tokyo", "paris"])
    #list(le.inverse_transform([2, 2, 1]))

    # 标签化
    lines = content.replace('\n','').split('。')
    lines = [i for i in lines if len(i)>1]
    lines_data = []

    # 词标签化
    X_data = []
    for line in lines:
        if len(line)==0:
            continue
        line_segs = jieba.cut(line)
        line_segs = [i for i in line_segs if len(i)>1]
        line_df = pd.DataFrame({'segment':line_segs})
        line_df = utils.remove_stopwords(line_df)
        line_data = leX.transform(line_df['segment'])
        X_data.append(line_data)

    # 每扩充为等长
    max_len = max(map(len, X_data))
    #print(max_len)
    X_data = [ np.append(l[:max_len], [blank_label]*(max_len-len(l))) for l in X_data ]

    # 句子标签规范化
    ley = preprocessing.LabelEncoder()
    ley.fit(lines)
    y_data = ley.transform(lines)

    return X_data, y_data, leX, ley, max_len
