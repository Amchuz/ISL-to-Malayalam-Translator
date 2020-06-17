#!/usr/bin/env python
# coding: utf-8

import cv2
import yaml

if True:  
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from lib.openpose import SkeletonDetector
    from lib.tracker import Tracker
    from lib.skeletons import ReadImagesTxt
    import lib.commons as commons


def par(path):  
    return ROOT + path if (path and path[0] != "/") else path

confg = commons.read_yaml(ROOT + "config/config.yaml")
con = confg["p1_get_skeletons.py"]

img_fname_format = confg["image_filename_format"]
skltn_fname_format = confg["skeleton_filename_format"]

if True:
    prgs_imgs_desc_txt = par(con["input"]["imgs_desc_txt"])
    prgs_imgs_folder = par(con["input"]["imgs_folder"])

if True:
    imgs_info_txt = par(con["output"]["imgs_info_txt"])
    detected_skltns_res = par(con["output"]["detected_skltns_res"])
    res_imgs_folder = par(con["output"]["res_imgs_folder"])

if True:
    model = con["openpose"]["model"]
    img_size = con["openpose"]["img_size"]

class ImageDisplayer(object):

    def __init__(self):
        self._window_name = "Window"
        cv2.namedWindow(self._window_name)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)


if __name__ == "__main__":

    skltn_detector = SkeletonDetector(model, img_size)
    multiperson_tracker = Tracker()

    images_loader = ReadImagesTxt(img_folder=prgs_imgs_folder, valid_imgs_txt=prgs_imgs_desc_txt,
                                                        img_filename_format=img_fname_format)

    images_loader.save_images_info(filepath=imgs_info_txt)
    img_displayer = ImageDisplayer()

    os.makedirs(os.path.dirname(imgs_info_txt), exist_ok=True)
    os.makedirs(detected_skltns_res, exist_ok=True)
    os.makedirs(res_imgs_folder, exist_ok=True)

    num_total_images = images_loader.num_images
    for ith_img in range(num_total_images):

        img, action_label, img_info = images_loader.read_image()
        humans = skeleton_detector.detect(img)
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.display(img_disp, wait_key_ms=1)
        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        mulperson = multiperson_tracker.track(skeletons)  
        save_skltns = [img_info + skeleton.tolist() for skeleton in mulperson.values()]
        filename = skltn_fname_format.format(ith_img)
        commons.save_listlist(detected_skltns_res + filename, save_skltns)

        filename = img_fname_format.format(ith_img)
        cv2.imwrite(res_imgs_folder + filename,img_disp)

        print(f"{ith_img}/{num_total_images} th image "
              f"has {len(skeletons)} people in it")

    print("Program ends")
