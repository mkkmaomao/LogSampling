import numpy as np
import pickle
import os
from collections import Counter
import pdb
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss, ClusterCentroids
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.decomposition import PCA

def CombineSampling(ws, percent, model):
    with open(r"D:/Python Projects/NeuralLog/data/embedding/{}/iforest-train-ws{}.pkl".format(dataset, ws),
              mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)

    o_ratio = int(y_tr.count(0)) / int(y_tr.count(1))

    ################# PCA ######################
    # pca = PCA(n_components=dim)  # 768->300
    # x_tr = np.array(x_tr)
    # print("x_tr shape: ", x_tr.shape)  # should be [xx,100,768]
    # x_PCA = []
    # for i in range(x_tr.shape[0]):
    #     x_PCA.append(pca.fit_transform(x_tr[i]))
    # x_PCA = np.array(x_PCA)
    # print("x_PCA shape: ", x_PCA.shape)  # should be [xx,100,50]
    # x_ex = np.reshape(x_PCA, (x_PCA.shape[0], -1))  # for sampling
    # print("x_ex shape: ", x_ex.shape)  # should be [xx,500]
    ################# PCA ######################

    x_tr_mean = np.mean(x_tr, axis=1)
    print("x_tr_mean.shape: ",x_tr_mean.shape)

    x_total, y_total = [], []
    print("==============The model is :", model)
    for pp in range(len(percent)):
        per = percent[pp]
        if model == "SMOTEENN":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = SMOTEENN(random_state=42, sampling_strategy=1 / (o_ratio * per))
            elif per == "auto":
                sm = SMOTEENN(random_state=42, sampling_strategy='auto')
            else:
                sm = SMOTEENN(random_state=42, sampling_strategy=1)

        elif model == "SMOTETomek":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = SMOTETomek(random_state=42, sampling_strategy=1 / (o_ratio * per))
            elif per == "auto":
                sm = SMOTETomek(random_state=42, sampling_strategy='auto')
            else:
                sm = SMOTETomek(random_state=42, sampling_strategy=1)

        elif model == "SMOTE":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = SMOTE(random_state=42, sampling_strategy=1 / (o_ratio * per))
            elif per == "auto":
                sm = SMOTE(random_state=42, sampling_strategy='auto')
            else:
                sm = SMOTE(random_state=42, sampling_strategy=1)

        elif model == "ADASYN":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = ADASYN(random_state=42, sampling_strategy=1 / (o_ratio * per))
            elif per == "auto":
                sm = ADASYN(random_state=42, sampling_strategy='auto')
            else:
                sm = ADASYN(random_state=42, sampling_strategy=1)

        elif model == "NearMiss":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = NearMiss(sampling_strategy=1 / (o_ratio * per))
            else:
                sm = NearMiss(sampling_strategy=1)

        elif model == "ClusterCentroids":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = ClusterCentroids(random_state=42, sampling_strategy=1 / (o_ratio * per))
            elif per == "auto":
                sm = ClusterCentroids(random_state=42, sampling_strategy='auto')
            else:
                sm = ClusterCentroids(random_state=42, sampling_strategy=1)

        x_tr_new, y_tr_new = sm.fit_resample(x_tr_mean,y_tr)
        print("x_tr_new.shape: ",x_tr_new.shape)
        x_tr_new = np.expand_dims(x_tr_new, axis=0)
        print("after expanding, x_tr_new.shape: ",x_tr_new.shape)
        x_total = np.repeat(x_tr_new, 100, 1)

        # x_tr_new, y_tr_new = sm.fit_resample(x_ex, y_tr)
        # x_total = np.reshape(x_tr_new, (x_tr_new.shape[0], -1, dim))
        print("x final shape: ", x_total.shape)
        y_total = y_tr_new
        print("Total samples: ", Counter(y_tr_new))

        import joblib
        with open("D:/Python Projects/NeuralLog/data/{}_PCA/{}/iforest-train-ws{}-per{}.pkl".format(model, dataset, ws,
                                                                                                    per),
                  mode="wb") as f:
            joblib.dump((x_total, y_total), f, protocol=pickle.HIGHEST_PROTOCOL)
        print("the file has been saved.")


def loadtest(ws):
    with open("D:/Python Projects/NeuralLog/data/embedding/{}/iforest-test-ws{}.pkl".format(dataset, ws),
              mode="rb") as f:
        (x_te, y_te) = pickle.load(f)

    x_tr_mean = np.mean(x_te, axis=1)
    x_tr_new = np.expand_dims(x_tr_mean, axis=0)
    x_te_total = np.repeat(x_tr_new, 100, 1)

    # x_PCA = []
    # pca = PCA(n_components=dim)
    # x_te = np.array(x_te)
    # for i in range(x_te.shape[0]):
    #     # print(x_te[i].shape)
    #     x_PCA.append(pca.fit_transform(x_te[i]))
    #
    # print("x_PCA shape: ", (np.array(x_PCA)).shape)  # should be [xx,100,50]

    print("the number of anomaly and normal samples in test set are: ",Counter(y_te))
    with open("D:/Python Projects/NeuralLog/data/{}_PCA/{}/iforest-test-ws{}.pkl".format(model, dataset, ws),
              mode="wb") as f:
        pickle.dump((x_te_total, y_te), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = [0, 0.25, 0.5, 0.75, 1]
    # p = 1
    # p = 'auto'
    ws = 100
    dataset = "Thunderbird"
    model = "SMOTE"
    dim = 100
    loadtest(ws=100)
    if p == 'auto':
        CombineSampling(ws=100, percent=p, model=model)
    else:
        CombineSampling(ws=100, percent=p, model=model)