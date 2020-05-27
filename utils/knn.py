"""
使用 K临近算法 knn
"""

import math, operator
from sklearn import neighbors
import os
import os.path
import pickle
import numpy as np
from . import prepare_data


# 训练
def train(train_file, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=True):

    X, y, leX, ley, X_max = prepare_data.load_data(train_file)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump( (knn_clf, leX, ley, X_max), f)

    return knn_clf


# 识别
def predict(X, knn_clf=None, model_path=None, distance_threshold=0.6):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf, leX, ley, X_max = pickle.load(f)

    # 扩充数组长度
    blank_label = leX.transform(['<blank>'])
    X = [i for i in X['segment'] if i in leX.classes_]
    X = leX.transform(X)
    X = np.append(X[:X_max], [blank_label]*(X_max-len(X)))

    # Use the KNN model to find the first 5 best matches for the test face
    # 返回5个最佳结果
    closest_distances = knn_clf.kneighbors([X], n_neighbors=5)

    #print(closest_distances)

    # 将阈值范围内的结果均返回
    l = knn_clf.classes_[knn_clf._y[closest_distances[1][0][0]]]
    return ley.inverse_transform([l])

