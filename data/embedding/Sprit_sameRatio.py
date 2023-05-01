import numpy as np
import pickle
from collections import Counter
def loadtrain():

    with open("./Spirit/iforest-train-ws100.pkl", mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)
    x_sum_tr = []
    y_sum_tr = []
    countA = 0
    countN = 0
    print(len(y_tr))
    for i in range(len(y_tr)-1,-1,-1):
        if (y_tr[i] == 0):
            if (countN < 20408): #20614*0.99
                x_sum_tr.append(x_tr[i])
                y_sum_tr.append(y_tr[i])
                countN = countN + 1
        else:
            if (countA < 19192):  #19385*0.99
                x_sum_tr.append(x_tr[i])
                y_sum_tr.append(y_tr[i])
                countA = countA + 1

    print("after the 1st loop (A: {} and N: {})".format(countA,countN))
    print("y_sum_tr: ",len(y_sum_tr))

    with open("./Spirit/iforest-train-ws100.pkl", mode="rb") as f:
        (x_tr2, y_tr2) = pickle.load(f)
    for j in range(50000,len(y_tr2)):
        if y_tr2[j] == 0:
            if countN < 20614:
                x_sum_tr.append(x_tr2[j])
                y_sum_tr.append(y_tr2[j])
                countN = countN + 1
        else:
            if countA < 19385:  #19385
                x_sum_tr.append(x_tr2[j])
                y_sum_tr.append(y_tr2[j])
                countA = countA + 1
    print("the number of anomaly and normal samples in training set are {} and {}. ".format(countA, countN))
    with open("./Spirit_sameratio/iforest-train-ws100.pkl", mode="wb") as f:
        pickle.dump((x_sum_tr, y_sum_tr), f, protocol=pickle.HIGHEST_PROTOCOL)



def loadtest():
    with open("Spirit_5m/iforest-test-ws100.pkl", mode="rb") as f:
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
    with open("./Spirit_sameratio/iforest-test-ws100.pkl", mode="wb") as f:
        pickle.dump((x_sum_te, y_sum_te), f, protocol=pickle.HIGHEST_PROTOCOL)

def gen_5m_train():
    with open("Spirit_5m/iforest-train-ws100.pkl", mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)
    x_sum_tr = []
    y_sum_tr = []
    countA = 0
    countN = 0
    print(len(y_tr))
    j=0
    for i in range(len(y_tr)-1,-1,-1):
        if (j<19999):
            x_sum_tr.append(x_tr[i])
            y_sum_tr.append(y_tr[i])
            j = j + 1
        # if (y_tr[i] == 0):
        #     if (countN < 20614): #20614*0.99
        #         x_sum_tr.append(x_tr[i])
        #         y_sum_tr.append(y_tr[i])
        #         countN = countN + 1
        # else:
        #     if (countA < 19385):  #19385*0.99
        #         x_sum_tr.append(x_tr[i])
        #         y_sum_tr.append(y_tr[i])
        #         countA = countA + 1

    print("Train: A: {} and N: {}".format(Counter(y_sum_tr)[1],Counter(y_sum_tr)[0]))
    with open("./Spirit_sameratio/iforest-train-ws100.pkl", mode="wb") as f:
        pickle.dump((x_sum_tr, y_sum_tr), f, protocol=pickle.HIGHEST_PROTOCOL)


def gen_5m_test():
    with open("./Spirit_5m/iforest-test-ws100.pkl", mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)
    x_sum_tr = []
    y_sum_tr = []
    countA = 0
    countN = 0

    # for i in range(40000, len(y_tr)):
    #     if (y_tr[i] == 0):
    #         if (countN < 9652):
    #             x_sum_tr.append(x_tr[i])
    #             y_sum_tr.append(y_tr[i])
    #             countN = countN + 1
    #     else:
    #         if (countA < 347):
    #             x_sum_tr.append(x_tr[i])
    #             y_sum_tr.append(y_tr[i])
    #             countA = countA + 1

    print("Test: A: {} and N: {}".format(y_tr.count(1),y_tr.count(0)))
    with open("./Spirit_sameratio/iforest-test-ws100.pkl", mode="wb") as f:
        pickle.dump((x_tr, y_tr), f, protocol=pickle.HIGHEST_PROTOCOL)

def load_embedding():
    with open("./Spirit_5m/iforest-train-ws100.pkl", mode="rb") as f:
        x_te, y_te = pickle.load(f)
    print(x_te[0:5])

if __name__ =="__main__":
    # loadtrain()
    # loadtest()
    gen_5m_train()
    gen_5m_test()
    # load_embedding()