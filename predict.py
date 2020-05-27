# -*- coding: utf-8 -*-

import os, sys
from datetime import datetime
from utils import knn
import jiebazhc as jieba
import pandas as pd
import utils

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

    line_segs = jieba.cut(test_thing)
    line_segs = [i for i in line_segs if len(i)>1]
    line_df = pd.DataFrame({'segment':line_segs})
    line_df = utils.remove_stopwords(line_df)

    # Using the trained classifier, make predictions for unknown text
    start_time = datetime.now()
    predictions = knn.predict(line_df, model_path=model_name, distance_threshold=1000)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    # Print results on the console
    print(predictions)
        