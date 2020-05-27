# -*- coding: utf-8 -*-

import os, sys
from utils import knn

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python3 %s <train_file> [model_name]" % sys.argv[0])
        sys.exit(2)

    train_data_dir = sys.argv[1]

    if len(sys.argv)>2:
        model_name = sys.argv[2]
    else:
        model_name = 'knn_model'

    # Train the KNN classifier and save it to disk
    print("Training KNN classifier...")
    classifier = knn.train(train_data_dir, model_save_path=model_name + '.clf', n_neighbors=2)
    print("Training complete!")
