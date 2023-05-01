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
from imblearn.under_sampling import NearMiss, ClusterCentroids, InstanceHardnessThreshold
from imblearn.combine import SMOTEENN, SMOTETomek

def CombineSampling(ws,percent,model):
    with open(r"D:/Python Projects/NeuralLog/data/embedding/{}/iforest-train-ws{}.pkl".format(dataset, ws),
              mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)

    o_ratio = int(y_tr.count(0))/int(y_tr.count(1))
    x_tr_mean = np.mean(x_tr, axis=1)

    x_total, y_total = [], []
    print("==============The model is :", model)
    if model == "SMOTEENN":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = SMOTEENN(random_state=42,sampling_strategy=1/(o_ratio*per))
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

    x_tr_new, y_tr_new = sm.fit_resample(x_tr_mean, y_tr)
    print("after" + model + "sampling, two class data changed:", Counter(y_tr_new))
    delete_new_index = []
    original_data_index = []

    for i in range(x_tr_mean.shape[0]):
        for j in range(x_tr_new.shape[0]):
            if np.array_equal(x_tr_mean[i], x_tr_new[j]):
                original_data_index.append(i)
                delete_new_index.append(j)

    kept_data, kept_y, add_data, add_y = [], [], [], []
    for i in range(x_tr_mean.shape[0]):
        if i in original_data_index:
            kept_data.append(x_tr[i])
            kept_y.append(y_tr[i])

    for j in range(x_tr_new.shape[0]):
        if j not in delete_new_index:
            add_data.append(x_tr_new[j])
            add_y.append(y_tr_new[j])

    kept_data = np.array(kept_data)
    add_data = np.array(add_data)
    add_data = np.expand_dims(add_data,axis=0)
    add_data = np.repeat(add_data, 100, 1)
    x_total = np.append(kept_data, add_data, axis=0)
    y_total = kept_y + add_y

    print("Total samples: ", Counter(y_total))
    print("Same as the y_tr_new:", Counter(y_tr_new))

    import joblib
    with open("D:/Python Projects/NeuralLog/data/{}_emb/{}/iforest-train-ws{}-per{}.pkl".format(model, dataset, ws,
                                                                                            percent),
          mode="wb") as f:
        joblib.dump((x_total, y_total), f, protocol=pickle.HIGHEST_PROTOCOL)



def Sampling(ws,percent,model):
    with open(r"D:/Python Projects/NeuralLog/data/embedding/{}/iforest-train-ws{}.pkl".format(dataset, ws),
              mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)

    o_ratio = int(y_tr.count(0))/int(y_tr.count(1))
    x_tr_mean = np.mean(x_tr, axis=1)

    x_total, y_total = [], []
    print("==============The model is :", model)
    if model == "SMOTE":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = SMOTE(random_state=42,sampling_strategy=1/(o_ratio*per))
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
                sm = NearMiss(sampling_strategy=1/(o_ratio*per))
            else:
                sm = NearMiss(sampling_strategy=1)

    elif model == "ClusterCentroids":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = ClusterCentroids(random_state=42, sampling_strategy=1/(o_ratio*per))
            elif per == "auto":
                sm = ClusterCentroids(random_state=42, sampling_strategy='auto')
            else:
                sm = ClusterCentroids(random_state=42, sampling_strategy=1)

    elif model == "InstanceHardnessThreshold":
            if per == 0.25 or per == 0.5 or per == 0.75:
                sm = ClusterCentroids(random_state=42, sampling_strategy=1/(o_ratio*per))
            elif per == "auto":
                sm = ClusterCentroids(random_state=42, sampling_strategy='auto')
            else:
                sm = ClusterCentroids(random_state=42, sampling_strategy=1)
        # print(len(x_tr))
        # print(len(x_tr[0]))
        # print(x_tr[0][0])

    x_tr_new, y_tr_new = sm.fit_resample(x_tr_mean,y_tr)

    if (len(x_tr) < len(x_tr_new)):
            x_tr_new = x_tr_new[len(x_tr):]
            y_tr_new = y_tr_new[len(y_tr):]
            print("after"+model+"sampling, the synthetic samples only has abnormal category:", Counter(y_tr_new))
            x_ex = np.expand_dims(x_tr_new, 1)
            x_times = np.repeat(x_ex, 100, 1) # e.g., T dataset: (82543,100,768)
            print(x_times.shape)

            x_total = np.append(x_tr, x_times, axis=0)
            y_total = np.append(y_tr, y_tr_new)
            print("Total samples: ", Counter(y_total))

    else:
            print("after" + model + "sampling, the normal samples has been deleted:", Counter(y_tr_new))
            kept_data_index = []
            for i in range(x_tr_new.shape[0]):  # undersample
                for j in range(x_tr_mean.shape[0]):  # original
                    if np.array_equal(x_tr_new[i], x_tr_mean[j]):
                        kept_data_index.append(j)
                        print("y_tr_new:", y_tr_new[i])
                        print("y_tr[j]", y_tr[j])

            print("after "+model+" sampling, the number of kept data is:", len(kept_data_index))

            for d in range(len(x_tr)):
                if d in kept_data_index:
                    x_total.append(x_tr[d])
                    y_total.append(y_tr[d])
            print("Total samples: ", Counter(y_total))


   # pdb.set_trace()
    import joblib
    with open("D:/Python Projects/NeuralLog/data/{}_emb/{}/iforest-train-ws{}-per{}.pkl".format(model,dataset, ws,
                                                                                                      percent),
              mode="wb") as f:
        joblib.dump((x_total, y_total), f, protocol=pickle.HIGHEST_PROTOCOL)


def loadtest(ws):
    with open("D:/Python Projects/NeuralLog/data/embedding/{}/iforest-test-ws{}.pkl".format(dataset,ws), mode="rb") as f:
        (x_te, y_te) = pickle.load(f)
    x_sum_te = []
    y_sum_te = []
    countA = 0
    countN = 0
    for i in range(len(y_te)):
        if y_te[i] == 0:
                x_sum_te.append(x_te[i])
                y_sum_te.append(y_te[i])
                countN = countN + 1
        else:
            x_sum_te.append(x_te[i])
            y_sum_te.append(y_te[i])
            countA = countA + 1

    print("the number of anomaly and normal samples in test set are {} and {}. ".format(countA, countN))
    with open("D:/Python Projects/NeuralLog/data/{}_emb/{}/iforest-test-ws{}.pkl".format(model,dataset,ws), mode="wb") as f:
        pickle.dump((x_sum_te, y_sum_te), f, protocol=pickle.HIGHEST_PROTOCOL)

def draw(model, dataset, percent):
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))

    with open(r"D:/Python Projects/NeuralLog/data/embedding/{}/iforest-train-ws{}.pkl".format(dataset, ws),
              mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)
    x_tr_old  = np.mean(x_tr, axis=1)

    with open("D:/Python Projects/NeuralLog/data/{}_emb/{}/iforest-train-ws{}-per{}.pkl".format(model,dataset, ws,
                                                                                                      percent),
              mode="rb") as f:
        (x_tr_new, y_tr_new) = pickle.load(f)

    newData = np.mean(x_tr_new,axis=1)
    newData['label'] = y_tr_new

    fig1 = sns.scatterplot(x=x_tr_old, y=y_tr_new, hue='label', ax=axs[0]).set(title='Imbalance data')
    fig2 = sns.scatterplot(data=newData, x='x', y='y', hue='label', ax=axs[1]).set(title='Balanced data(SMOTE)')
    scatter_fig1 = fig1.get_figure()
    scatter_fig2 = fig2.get_figure()
    scatter_fig1.savefig("D:/Python Projects/NeuralLog/data/{}_emb/{}/iforest-train-ws{}-per{}-originaldata.png".
                         format(model,dataset, ws,percent), dpi=400)
    scatter_fig2.savefig("D:/Python Projects/NeuralLog/data/{}_emb/{}/iforest-train-ws{}-per{}-resampleddata.png".
                         format(model,dataset, ws,percent), dpi=400)


if __name__ == "__main__":
    p = [0,0.25,0.5,0.75]
    # p = 'auto'
    ws=100
    dataset = "Thunderbird"
    model = "InstanceHardnessThreshold"
    if p == 'auto':
        per = p
        if model == "SMOTEENN" or model == "SMOTETomek":
            CombineSampling(ws=100,percent=per,model=model)
        else:
            Sampling(ws=100,percent=per, model=model)
        loadtest(ws=100)
    else:
        for i in range(len(p)):
            per = p[i]
            if model == "SMOTEENN" or model == "SMOTETomek":
                CombineSampling(ws=100, percent=per, model=model)
            else:
                Sampling(ws=100,percent=per,model=model)
            loadtest(ws=100)
            #draw(model,dataset,percent=per)