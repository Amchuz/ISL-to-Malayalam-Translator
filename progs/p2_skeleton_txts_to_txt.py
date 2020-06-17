#!/usr/bin/env python
# coding: utf-8

import numpy as np
import simplejson
import collections

if True: 
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import lib.commons as commons


def par(path):
    return ROOT + path if (path and path[0] != "/") else path

confg = commons.read_yaml(ROOT + "config/config.yaml")
con = confg["p2_skeleton_txts_to_txt.py"]

CLASSES = np.array(confg["classes"])

skltn_fname_format = confg["skeleton_filename_format"]

detected_skltns_folder = par(con["input"]["detected_skltns_folder"])
skltns_txt = par(con["output"]["skltns_txt"])

first_person = 0  
action_label = 3 

def read_skltns_txt(i):

    filename = detected_skltns_folder + \
        skltn_fname_format.format(i)
    skltns_in_ith_txt = commons.read_listlist(filename)
    return skltns_in_ith_txt


def get_length(filepaths):
  
    for i in range(len(filepaths)):
        skeletons = read_skltns_txt(i)
        if len(skeletons):
            skeleton = skeletons[first_person]
            data_size = len(skeleton)
            assert(data_size == 41)
            return data_size
    raise RuntimeError(f"No valid txt under: {detected_skltns_folder}.")


if __name__ == "__main__":

    filepaths = commons.get_filenames(detected_skltns_folder,
                                          use_sort=True, with_folder_path=True)
    num_skltns = len(filepaths)

    data_length = get_length(filepaths)
    print("Data length of one skeleton is {data_length}")

    all_skeletons = []
    labels_cnt = collections.defaultdict(int)
    for i in range(num_skltns):

        
        skeletons = read_skltns_txt(i)
        if not skeletons: 
            continue
        skeleton = skeletons[first_person]
        label = skeleton[action_label]
        if label not in CLASSES:  
            continue
        labels_cnt[label] += 1

        
        all_skeletons.append(skeleton)

        if i == 1 or i % 100 == 0:
            print("{}/{}".format(i, num_skltns))

    with open(skltns_txt, 'w') as f:
        simplejson.dump(all_skeletons, f)

    print(f"There are {len(all_skeletons)} skeleton data.")
    print(f"They are saved to {skltns_txt}")
    print("Number of each action: ")
    for label in CLASSES:
        print(f"    {label}: {labels_cnt[label]}")
