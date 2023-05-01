import os
import sys
sys.path.append("../")
from sklearn.metrics import classification_report
import pickle
import numpy as np
import pdb
from tensorflow.keras.utils import Sequence
# from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from official.nlp import optimization
from sklearn.utils import shuffle
from collections import Counter
import tensorflow_addons as tfa
from neurallog.models import transformers
from neurallog import data_loader
import tensorflow
from tensorflow.python.client import device_lib
import joblib
class BatchGenerator(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print(self.batch_size)
        dummy = np.zeros(shape=(embed_dim,))
        x = self.X[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.X))]
        X = np.zeros((len(x), max_len, embed_dim))
        Y = np.zeros((len(x), 2))
        item_count = 0
        for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.X))):
            x = self.X[i]
            if len(x) > max_len:
                x = x[-max_len:]
            x = np.pad(np.array(x), pad_width=((max_len - len(x), 0), (0, 0)), mode='constant',
                       constant_values=0)
            # print(x[item_count].shape)
            # pdb.set_trace()
            X[item_count] = np.reshape(x, [max_len, embed_dim])
            Y[item_count] = self.Y[i]
            item_count += 1

            # print("x shape: ", np.shape(X))
            # print("y shape: ", np.shape(Y))
            # print("y: ", Y)

        return X[:], Y[:, 0]
        # return X[:], Y[:]


def train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                    epoch_num, model_name=None):

    epochs = epoch_num
    steps_per_epoch = num_train_samples
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    loss_object = SparseCategoricalCrossentropy()
    # triplet_loss
    # loss_object = tfa.losses.TripletSemiHardLoss()

    model = transformers.transformer_classifer(embed_dim, 2048, 75, 12, loss_object, optimizer)
    # modified 768->500
    # model.load_weights("hdfs_transformer.hdf5")

    print(model.summary())

    # checkpoint
    filepath = model_name
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=True)
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min',
        baseline=None, restore_best_weights=True
    ) # monitor='val_accuracy', mode='auto'
    callbacks_list = [checkpoint, early_stop]

    # print("before fit_generator")
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=int(num_train_samples / batch_size),
                        epochs=epoch_num,
                        verbose=1,
                        validation_data=validate_generator,
                        validation_steps=int(num_val_samples / batch_size),
                        workers=16,
                        max_queue_size=32,
                        callbacks=callbacks_list,
                        shuffle=True
                        )
    # print("after fit_generator")
    return model


def train(X, Y, epoch_num, batch_size, model_file=None):
    X, Y = shuffle(X, Y)
    n_samples = len(X)
    train_x, train_y = X[:int(n_samples * 90 / 100)], Y[:int(n_samples * 90 / 100)]
    val_x, val_y = X[int(n_samples * 90 / 100):], Y[int(n_samples * 90 / 100):]

    training_generator, num_train_samples = BatchGenerator(train_x, train_y, batch_size), len(train_x)
    validate_generator, num_val_samples = BatchGenerator(val_x, val_y, batch_size), len(val_x)

    print("Number of training samples: {0} - Number of validating samples: {1}".format(num_train_samples,
                                                                                       num_val_samples))

    # print("the shape after batchGenerator: ", training_generator)
    model = train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                            epoch_num, model_name=model_file)

    return model

def test_model(x, y, batch_size, epochs):
    x, y = shuffle(x, y)
    x, y = x[: len(x) // batch_size * batch_size], y[: len(y) // batch_size * batch_size]
    test_loader = BatchGenerator(x, y, batch_size)

    steps_per_epoch = len(x)
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    loss_object = SparseCategoricalCrossentropy()
    # triplet loss:
    # loss_object = tfa.losses.TripletSemiHardLoss()

    model = transformers.transformer_classifer(embed_dim, 2048, 75, 12, loss_object, optimizer)
    # modified 768->500 PCA
    model.load_weights(save_file)
    prediction = model.predict(test_loader, steps=(len(x) // batch_size), workers=16, max_queue_size=32,
                                         verbose=1)
    prediction = np.argmax(prediction, axis=1)
    y = y[:len(prediction)]
    y_true = np.argmax(test_loader.Y, axis=1)
    print("y length: ", len(y_true))
    print("prediction: ", prediction)
    print("predictions: ", Counter(prediction))
    print("true labels: ", Counter(y_true))

    # report = classification_report(np.array(y), prediction)
    # print(report)
    def calculate_metric():
        TP, TN, FN, FP = 0, 0, 0, 0
        for i in range(len(prediction)):
            if (prediction[i] == y_true[i]):
                if (prediction[i] == 1):
                    TP = TP + 1
                else:
                    TN = TN + 1
            else:
                if ((prediction[i] == 1) & (y_true[i] == 0)):
                    FP = FP + 1
                elif ((prediction[i] == 0) & (y_true[i] == 1)):
                    FN = FN + 1

        Recall = TP / (TP + FN)
        Precision = TP / (TP + FP)
        Spec = TN / (TN + FP)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        print("tp: ", TP)
        print("fn: ", FN)
        print("fp: ", FP)
        print("Recall: ", Recall)
        print("Precision: ", Precision)
        print("Spec: ",Spec)
        print("F1: ", F1)
        return F1

    F1 = calculate_metric()
    return F1

def load_test(x_te, y_te):

    binary_array = [] # p(normal, abnormalous)
    for i in range(len(y_te)):
        # print("y_te[i]", y_te[i])
        if (y_te[i] == 0):
            binary_array.append((1, 0))
        else:
            binary_array.append((0, 1))

    # print("binary array: ", binary_array)
    return x_te, binary_array

if __name__ == '__main__':

    ws = "100"
    epoch = 20
    dataset = 'Thunderbird'
    samplingmethod = 'SMOTE_PCA'
    per = 0.5
    # log_file = "../data/raw/BGL.log"
    emb_dir = "/home/xiaoxuema3/NeuralLog/data/{}/{}".format(samplingmethod,dataset)
    embed_dim = 100  # Embedding size for each token 100*100->PCA 768->original
    max_len = 100

    # print("aaaaaaaaaaaaaaaa")
    # print(tensorflow.__version__)
    print(device_lib.list_local_devices())

    # (x_tr, y_tr), (x_te, y_te) = data_loader.load_Supercomputers(
    #     log_file, train_ratio=0.8, windows_size=20,
    #     step_size=5, e_type='bert', e_name=None, mode='balance')

    save_file = "../SamplingResults/{}/{}/ws={}_per={}_epoch{}.hdf5".format(samplingmethod, dataset, ws, per, epoch)
    # with open(os.path.join(emb_dir, "iforest-train-ws{}.pkl".format(ws)), mode="rb") as f:
    # save_file = "../results_tripletloss/{}/hdfs_transformer_epoch{}.hdf5".format(dataset, epoch)
    with open("../data/{}/{}/iforest-train-ws{}-per{}.pkl".format(samplingmethod,dataset, ws, per), mode="rb") as f:
    # with open("../data/embedding/{}/iforest-train-wssession.pkl".format(dataset), mode="rb") as f:
        (x_tr, y_tr) = joblib.load(f)
    print(Counter(y_tr))
    x_tr = np.asarray(x_tr)
    y_tr = np.asarray(y_tr)

    # print("x_tr :", np.shape(x_tr))
    with open("../data/{}/{}/iforest-test-ws{}.pkl".format(samplingmethod,dataset,ws), mode="rb") as f:
    # with open("../data/embedding/{}/iforest-test-wssession.pkl".format(dataset), mode="rb") as f:
        (x_te, y_te) = pickle.load(f)
    print(Counter(y_te))
    # x_te = np.asarray(x_te)
    # y_te = np.asarray(y_te)
    x_test, y_test = load_test(x_te, y_te)

    model = train(x_tr, y_tr, epoch, 64, save_file)
    test_model(x_test, y_test, batch_size=64, epochs=epoch)

