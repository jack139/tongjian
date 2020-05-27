# -*- coding: utf-8 -*-

import jiebazhc as jieba
import numpy as np
import codecs
import pandas as pd
import matplotlib.pyplot as plt
#from wordcloud import WordCloud

file = codecs.open("data/001.txt", 'r', 'utf-8')
content = file.read()
file.close()

segments = []
segs = jieba.cut(content)

for seg in segs:
    if len(seg)>1:
        segments.append(seg)

segmentDF = pd.DataFrame({'segment':segments})

# 停用词
stopwords = pd.read_csv("data/cn_stopwords.txt", encoding='utf8', index_col=False, quoting=3, sep="\n")
segmentDF = segmentDF[~segmentDF.segment.isin(stopwords.stopword)]

# 文言虚词
wyStopWords = pd.Series([
    # 42 个文言虚词
    '之', '其', '或', '亦', '方', '于', '即', '皆', '因', '仍', '故',
    '尚', '呢', '了', '的', '着', '一', '不', '乃', '呀', '吗', '咧',
    '啊', '把', '让', '向', '往', '是', '在', '越', '再', '更', '比',
    '很', '偏', '别', '好', '可', '便', '就', '但', '儿',
    # 高频副词
    '又', '也', '都', '要',
    # 高频代词
    '这', '那', '你', '我', '他',
    #高频动词
    '来', '去', '道', '笑', '说',
    #空格
    ' ', ''
]);

segmentDF = segmentDF[~segmentDF.segment.isin(wyStopWords)]

#对词频进行统计。

segStat = segmentDF.groupby(by=["segment"])["segment"].agg(count="count").reset_index().sort_values(by=['count'],ascending=False)

print(segStat.head(100))


