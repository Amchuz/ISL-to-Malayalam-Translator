#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import classification_report

if True: 
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import lib.plot as plot
    import lib.commons commons
    from lib.classifier import ClassifierOfflineTrain

def par(path): 
    return ROOT + path if (path and path[0] != "/") else path

confg = commons.read_yaml(ROOT + "config/config.yaml")
con = confg["p4_train.py"]

CLASSES = np.array(confg["classes"])


processed_features = par(con["input"]["processed_features"])
processed_feature_labels = par(con["input"]["processed_features_labels"])

model_path= par(con["output"]["model_path"])


def train_test_split(X, Y, ratio_of_test_size):

    IS_SPLIT_BY_SKLEARN_FUNC = True

    if IS_SPLIT_BY_SKLEARN_FUNC:
        RAND_SEED = 1
        tr_X, te_X, tr_Y, te_Y = sklearn.model_selection.train_test_split(
            X, Y, test_size=ratio_of_test_size, random_state=RAND_SEED)
    else:
        tr_X = np.copy(X)
        tr_Y = Y.copy()
        te_X = np.copy(X)
        te_Y = Y.copy()
    return tr_X, te_X, tr_Y, te_Y

def evaluate_model(model, classes, tr_X, tr_Y, te_X, te_Y):

    t0 = time.time()

    tr_accu, tr_Y_predict = model.predict_and_evaluate(tr_X, tr_Y)
    print(f"Accuracy on training set is {tr_accu}")

    te_accu, te_Y_predict = model.predict_and_evaluate(te_X, te_Y)
    print(f"Accuracy on testing set is {te_accu}")

    print("Accuracy report:")
    print(classification_report(
        te_Y, te_Y_predict, target_names=classes, output_dict=False))

    average_time = (time.time() - t0) / (len(tr_Y) + len(te_Y))
    print("Time cost for predicting one sample: "
          "{:.5f} seconds".format(average_time))

    axis, cf = plot.plot_confusion_matrix(
        te_Y, te_Y_predict, classes, normalize=False, size=(12, 8))
    plt.show()


def main():


    print("\nReading csv files of classes, features, and labels ...")
    X = np.loadtxt(processed_features, dtype=float)  
    Y = np.loadtxt(processed_features_labels, dtype=int) 

    tr_X, te_X, tr_Y, te_Y = train_test_split(
        X, Y, ratio_of_test_size=0.3)
    print("\nAfter train-test split:")
    print("Size of training data X:    ", tr_X.shape)
    print("Number of training samples: ", len(tr_Y))
    print("Number of testing samples:  ", len(te_Y))

    print("\nStart training model ...")
    model = ClassifierOfflineTrain()
    model.train(tr_X, tr_Y)

    print("\nStart evaluating model ...")
    evaluate_model(model, CLASSES, tr_X, tr_Y, te_X, te_Y)

    print("\nSave model to " + model_path)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
