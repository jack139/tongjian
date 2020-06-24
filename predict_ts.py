# -*- coding: utf-8 -*-

import os, sys
from datetime import datetime
from utils import knn
import jiebazhc as jieba
import pandas as pd
import utils

X_BLANK = 'blank'


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python3 %s <test txt> [model_name]" % sys.argv[0])
        sys.exit(2)


    test_thing = sys.argv[1]

    if len(sys.argv)>2:
        model_name = sys.argv[2]
    else:
        model_name = 'knn_model'

    if not model_name.endswith('.clf'):
        model_name += '.clf'

    all_y = []

    #line_segs = jieba.cut(test_thing)
    #line_segs = [i for i in line_segs if len(i)>0]
    line_segs = test_thing.split()

    for i in range(len(line_segs)):
        # 前一个词
        if i==0:
            last_word = X_BLANK
        else:
            last_word = line_segs[i-1]

        # 后一个词
        if i+1==len(line_segs):
            next_word = X_BLANK
        else:
            next_word = line_segs[i+1]

        # 词特征标签
        word_feature = [line_segs[i], last_word, next_word]
        print(word_feature)
        word_df = pd.DataFrame({'segment':word_feature})

        # Using the trained classifier, make predictions for unknown text
        start_time = datetime.now()
        predictions = knn.predict(word_df, model_path=model_name, distance_threshold=1000)
        print('[Time taken: {!s}]'.format(datetime.now() - start_time))

        # Print results on the console
        all_y.append(predictions[0])
        
    print(''.join(all_y))