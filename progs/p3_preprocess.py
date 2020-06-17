#!/usr/bin/env python
# coding: utf-8

import numpy as np

if True: 
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import lib.commons as commons
    from lib.skeletons import load_skeleton_data
    from lib.feature import extract_multi_frame_features


def par(path):  
    return ROOT + path if (path and path[0] != "/") else path

confg = commons.read_yaml(ROOT + "config/config.yaml")
con = confg["p3_preprocess.py"]

CLASSES = np.array(confg["classes"])

window_size = int(confg["features"]["window_size"]) 

skltns_txt = par(cfg["input"]["skltns_txt"])
processed_features = par(cfg["output"]["processed_features"])
processed_features_labels = par(cfg["output"]["processed_features_labels"])

def process_features(X0, Y0, indices, classes):

    add_noise = False
    if add_noise:
        X1, Y1 = extract_multi_frame_features(
            X0, Y0, indices, window_size, 
            is_adding_noise=True, is_print=True)
        X2, Y2 = extract_multi_frame_features(
            X0, Y0, indices, window_size,
            is_adding_noise=False, is_print=True)
        X = np.vstack((X1, X2))
        Y = np.concatenate((Y1, Y2))
        return X, Y
    else:
        X, Y = extract_multi_frame_features(
            X0, Y0, indices, window_size
            is_adding_noise=False, is_print=True)
        return X, Y

def main():

    X0, Y0, indices = load_skeleton_data(skltns_txt, CLASSES)

    print("\nExtracting time-serials features ...")
    X, Y = process_features(X0, Y0, indices, CLASSES)
    print(f"X.shape = {X.shape}, len(Y) = {len(Y)}")

    print("\nWriting features and labesl to disk ...")

    os.makedirs(os.path.dirname(processed_features), exist_ok=True)
    os.makedirs(os.path.dirname(processed_features_labels), exist_ok=True)

    np.savetxt(processed_features, X, fmt="%.5f")
    print("Save features to: " + processed_features)

    np.savetxt(processed_features_labels, Y, fmt="%i")
    print("Save labels to: " + processed_features_labels)


if __name__ == "__main__":
    main()
