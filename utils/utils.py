# -*- coding: utf-8 -*-
import codecs
import pandas as pd


# 停用词字典
stopwords = pd.read_csv("utils/cn_stopwords.txt", encoding='utf8', index_col=False, quoting=3, sep="\n")
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

# 去除停用词
def remove_stopwords(segmentDF):
    segmentDF = segmentDF[~segmentDF.segment.isin(stopwords.stopword)]
    segmentDF = segmentDF[~segmentDF.segment.isin(wyStopWords)]
    return segmentDF

def load_content(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as file:
        content = file.read()
    return content

def load_content_lines(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as file:
        content = file.readlines()
    return content