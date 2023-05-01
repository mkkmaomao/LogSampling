import numpy as np
import pickle
import os
from collections import Counter
def loadtrain(ws, percent):
    print (os.getcwd())
    with open(r"D:/Python Projects/NeuralLog/data/embedding/{}/iforest-train-ws{}-last.pkl".format(dataset,ws), mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)
    x_sum_tr = []
    y_sum_tr = []
    countN = 0
    print(len(y_tr))
    # y_tr = y_tr.tolist()
    print(y_tr.count(1))

    import random

    def shuffle(data, label):
        new_data = []
        new_label = []
        index = [i for i in range(len(data))]
        random.seed(0)
        random.shuffle(index)
        for ii in range(len(index)):
            new_data.append(data[index[ii]])
            new_label.append(label[index[ii]])
        return new_data, new_label

    x_tr,y_tr = shuffle(x_tr,y_tr)

    for i in range(len(y_tr) - 1, -1, -1):
       if (y_tr[i] == 0):
           if (countN < (y_tr.count(0) * percent)):
               x_sum_tr.append(x_tr[i])
               y_sum_tr.append(y_tr[i])
               countN = countN + 1
           else:
               pass
       else:
           x_sum_tr.append(x_tr[i])
           y_sum_tr.append(y_tr[i])


    def ratio_sampling(x_tr,y_tr,x_sum_tr,y_sum_tr, percent):
        for i in range(len(y_tr) - 1, -1, -1):
            if (y_tr[i] == 0):
                if (countN < (y_tr.count(1) * percent)):
                    x_sum_tr.append(x_tr[i])
                    y_sum_tr.append(y_tr[i])
                    countN = countN + 1
                else:
                    pass
            else:
                x_sum_tr.append(x_tr[i])
                y_sum_tr.append(y_tr[i])
        return x_sum_tr,y_sum_tr



    print("after the 1st loop (A: {} and N: {})".format(y_sum_tr.count(1),y_sum_tr.count(0)))
    print("y_sum_tr: ",len(y_sum_tr))

    with open("D:/Python Projects/NeuralLog/data/Undersampling_emb/{}/iforest-train-ws{}-per{}.pkl".format(dataset,ws, percent), mode="wb") as f:
        pickle.dump((x_sum_tr, y_sum_tr), f, protocol=pickle.HIGHEST_PROTOCOL)



def loadtest(ws):
    with open("D:/Python Projects/NeuralLog/data/embedding/{}/iforest-test-ws{}-last.pkl".format(dataset,ws), mode="rb") as f:
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
    with open("D:/Python Projects/NeuralLog/data/Undersampling_emb/{}/iforest-test-ws{}.pkl".format(dataset,ws), mode="wb") as f:
        pickle.dump((x_sum_te, y_sum_te), f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    p = [0.25,0.5,0.75,1]
    dataset = "Spirit_5m"
    for i in range(len(p)):
        per = p[i]
        loadtrain(ws=100,percent=per)
        loadtest(ws=100)