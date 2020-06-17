import numpy as np
import cv2
import os
import sys
import simplejson
from sklearn.preprocessing import OneHotEncoder

LEN_IMG_INFO = 5
LEN_SKELETON_XY = 18*2
NaN = 0 

def get_training_imgs_info(
        valid_images_txt,
        img_filename_format="{:05d}.jpg"):

    images_info = list()

    with open(valid_images_txt) as f:

        folder_name = None
        action_label = None
        cnt_action = 0
        actions = set()
        action_images_cnt = dict()
        cnt_clip = 0
        cnt_image = 0

        for cnt_line, line in enumerate(f):

            if line.find('_') != -1: 
                folder_name = line[:-1]
                action_label = folder_name.split('_')[0]
                if action_label not in actions:
                    cnt_action += 1
                    actions.add(action_label)
                    action_images_cnt[action_label] = 0

            elif len(line) > 1:  
                indices = [int(s) for s in line.split()]
                idx_start = indices[0]
                idx_end = indices[1]
                cnt_clip += 1
                for i in range(idx_start, idx_end+1):
                    filepath = folder_name+"/" + img_filename_format.format(i)
                    cnt_image += 1
                    action_images_cnt[action_label] += 1

                    image_info = [cnt_action, cnt_clip,
                                  cnt_image, action_label, filepath]
                    assert(len(image_info) == LEN_IMG_INFO)
                    images_info.append(image_info)

        print("")
        print("Number of action classes = {}".format(len(actions)))
        print("Number of training images = {}".format(cnt_image))
        print("Number of training images of each action:")
        for action in actions:
            print("  {:>8}| {:>4}|".format(
                action, action_images_cnt[action]))

    return images_info


class ReadImagesTxt(object):

    def __init__(self, img_folder, valid_imgs_txt,
                 img_filename_format="{:05d}.jpg"):
 
        self.images_info = get_training_imgs_info(
            valid_imgs_txt, img_filename_format)
        self.imgs_path = img_folder
        self.i = 0
        self.num_images = len(self.images_info)
        print(f"Reading images from txtscript: {img_folder}")
        print(f"Reading images information from: {valid_imgs_txt}")
        print(f"    Num images = {self.num_images}\n")

    def save_images_info(self, filepath):
        folder_path = os.path.dirname(filepath)
        os.makedirs(folder_path, exist_ok=True)
        with open(filepath, 'w') as f:
            simplejson.dump(self.images_info, f)

    def read_image(self):
        self.i += 1
        if self.i > len(self.images_info):
            raise RuntimeError(f"There are only {len(self.images_info)} images, "
                               f"but you try to read the {self.i}th image")
        filepath = self.get_filename(self.i)
        img = self.imread(self.i)
        if img is None:
            raise RuntimeError("The image file doesn't exist: " + filepath)
        img_action_label = self.get_action_label(self.i)
        img_info = self.get_image_info(self.i)
        return img, img_action_label, img_info

    def imread(self, index):
        return cv2.imread(self.imgs_path + self.get_filename(index))

    def get_filename(self, index):

        return self.images_info[index-1][4]

    def get_action_label(self, index):

        return self.images_info[index-1][3]

    def get_image_info(self, index):
        return self.images_info[index-1]


def load_skeleton_data(filepath, classes):

    label2index = {c: i for i, c in enumerate(classes)}

    with open(filepath, 'r') as f:

        dataset = simplejson.load(f)

        def is_good_data(row):
            return row[0] != 0
        dataset = [row for i, row in enumerate(dataset) if is_good_data(row)]

        X = np.array([row[LEN_IMG_INFO:LEN_IMG_INFO+LEN_SKELETON_XY]
                      for row in dataset])

        video_indices = [row[1] for row in dataset]
        Y_str = [row[3] for row in dataset]
        Y = [label2index[label] for label in Y_str]
        if 0:
            valid_indices = _get_skeletons_with_complete_upper_body(X, NaN)
            X = X[valid_indices, :]
            Y = [Y[i] for i in valid_indices]
            video_indices = [video_indices[i] for i in valid_indices]
            print("Num samples after removal = ", len(Y))
        N = len(Y)
        P = len(X[0])
        C = len(classes)
        print(f"\nNumber of samples = {N} \n"
              f"Raw feature length = {P} \n"
              f"Number of classes = {C}")
        print(f"Classes: {classes}")

        return X, Y, video_indices

    raise RuntimeError("Failed to load skeletons txt: " + filepath)


def _get_skeletons_with_complete_upper_body(X, NaN=0):

    left_idx, right_idx = 0, 14 * 2  

    def is_valid(x):
        return len(np.where(x[left_idx:right_idx] == NaN)[0]) == 0
    valid_indices = [i for i, x in enumerate(X) if is_valid(x)]
    return valid_indices

