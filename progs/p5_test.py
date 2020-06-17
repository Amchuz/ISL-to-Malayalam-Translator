#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import argparse
if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import lib.io as lib_images_io
    import lib.plot as plot
    import lib.commons as commons
    from lib.openpose import SkeletonDetector
    from lib.tracker import Tracker
    from lib.classifier import ClassifierOnlineTest
    from lib.classifier import *  


def par(path): 
    return ROOT + path if (path and path[0] != "/") else path


def get_command_line_arguments():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Test action recognition on \n"
            "(1) a video, (2) a folder of images, (3) or web camera.")
        parser.add_argument("-m", "--model_path", required=False,
                            default='model/trained_classifier.pickle')
        parser.add_argument("-t", "--data_type", required=False, default='webcam',
                            choices=["video", "folder", "webcam"])
        parser.add_argument("-p", "--data_path", required=False, default="",
                            help="path to a video file, or images folder, or webcam. \n"
                            "For video and folder, the path should be "
                            "absolute or relative to this project's root. "
                            "For webcam, either input an index or device name. ")
        parser.add_argument("-o", "--output_folder", required=False, default='output/',
                            help="Which folder to save result to.")

        args = parser.parse_args()
        return args
    args = parse_args()
    if args.data_type != "webcam" and args.data_path and args.data_path[0] != "/":
        args.data_path = ROOT + args.data_path
    return args


def get_folder_name(data_type,data_path):

    assert(data_type in ["video", "folder", "webcam"])

    if data_type == "video":  
        folder_name = os.path.basename(data_path).split(".")[-2]

    elif data_type == "folder":  
        folder_name = data_path.rstrip("/").split("/")[-1]

    elif data_type == "webcam":
        folder_name = lib_commons.get_time_string()

    return folder_name


args = get_command_line_arguments()

data_type = args.data_type
data_path = args.data_path
model_path = "//home/amchuz/Desktop/Project/progs/lib/model/trained_classifier.pickle"

folder_name = get_folder_name(data_type, data_path)

confg = commons.read_yaml("/home/amchuz/Desktop/Project/progs/lib/config/config.yaml")
con = confg["p5_test.py"]

CLASSES = np.array(confg["classes"])
skltn_fname_format = confg["skeleton_filename_format"]

window_size = int(confg["features"]["window_size"])

folder = args.output_folder + "/" + folder_name + "/"
skltn_folder_name = con["output"]["skeleton_folder_name"]
video_name = con["output"]["video_name"]

video_fps = float(con["output"]["video_fps"])

webcam_fps = float(con["settings"]["source"]
                           ["webcam_max_framerate"])

sample_interval = int(con["settings"]["source"]
                                ["video_sample_interval"])
model = con["settings"]["openpose"]["model"]
img_size = con["settings"]["openpose"]["img_size"]

desired_rows = int(con["settings"]["display"]["desired_rows"])

def select_images_loader(data_type, data_path):
    if data_type == "video":
        images_loader = lib_images_io.ReadFromVideo(
            data_path,
            sample_interval=sample_interval)

    elif data_type == "folder":
        images_loader = lib_images_io.ReadFromFolder(
            folder_path=data_path)

    elif data_type == "webcam":
        if data_path == "":
            webcam_idx = 0
        elif data_path.isdigit():
            webcam_idx = int(data_path)
        else:
            webcam_idx = data_path
        images_loader = lib_images_io.ReadFromWebcam(
            webcam_fps, webcam_idx)
    return images_loader


class MultiPersonClassifier(object):

    def __init__(self, model_path, classes):

        self.mulperson = {} 
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, window_size, human_id)

    def classify(self, mulperson_skltn):
   
        old_ids = set(self.mulperson)
        cur_ids = set(mulperson_skltn)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.mulperson[human]

        label = {}
        for id, skeleton in mulperson_skltn.items():

            if id not in self.mulperson:  
                self.mulperson[id] = self._create_classifier(id)

            classifier = self.mulperson[id]
            label[id] = classifier.predict(skeleton)  

        return label

    def get_classifier(self, id):
        if len(self.mulperson) == 0:
            return None
        if id == 'min':
            id = min(self.mulperson.keys())
        return self.mulperson[id]


def remove_skeletons_with_few_joints(skeletons):
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)

        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            good_skeletons.append(skeleton)
    return good_skeletons


def draw_result_img(img_disp, ith_img, humans, mulperson_skltn,
                    skeleton_detector, multiperson_classifier):

    r, c = img_disp.shape[0:2]
    desired_cols = int(1.0 * c * (desired_rows / r))
    img_disp = cv2.resize(img_disp,
                          dsize=(desired_cols, desired_rows))

    skeleton_detector.draw(img_disp, humans)

    if len(mulperson_skltn):
        for id, label in mulperson_label.items():
            skeleton = mulperson_skltn[id]
            skeleton[1::2] = skeleton[1::2] / scale_h           
            plot.draw_action_result(img_disp, id, skeleton, label)
    img_disp = plot.add_white_region_to_left_of_image(img_disp)

    cv2.putText(img_disp, "Frame:" + str(ith_img),
                (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN,
                color=(0, 0, 0), thickness=2)

    if len(mulperson_skltn):
        classifier_of_a_person = multiperson_classifier.get_classifier(
            id='min')
        classifier_of_a_person.draw_scores_onto_image(img_disp)
    return img_disp


def get_the_skeleton_data_to_save_to_disk(mulperson_skltn):
    skels_to_save = []
    for human_id in mulperson_skltn.keys():
        label = mulperson_label[human_id]
        skeleton = mulperson_skltn[human_id]
        skels_to_save.append([[human_id, label] + skeleton.tolist()])
    return skels_to_save


if __name__ == "__main__":


    skeleton_detector = SkeletonDetector(model, img_size)

    multiperson_tracker = Tracker()

    multiperson_classifier = MultiPersonClassifier(model_path, CLASSES)

    images_loader = select_images_loader(data_type, data_path)
    img_displayer = lib_images_io.ImageDisplayer()

    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder + skltn_folder_name, exist_ok=True)

    video_writer = lib_images_io.VideoWriter(
        folder + video_name, video_fps)

    try:
        ith_img = -1
        while images_loader.has_image():

            img = images_loader.read_image()
            ith_img += 1
            img_disp = img.copy()
            print(f"\nProcessing {ith_img}th image ...")

            humans = skeleton_detector.detect(img)
            skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
            skeletons = remove_skeletons_with_few_joints(skeletons)

            mulperson_skltn = multiperson_tracker.track(skeletons)  

            if len(mulperson_skltn):
                mulperson_label = multiperson_classifier.classify(mulperson_skltn)

            img_disp = draw_result_img(img_disp, ith_img, humans, mulperson_skltn,
                                       skeleton_detector, multiperson_classifier)

            if len(mulperson_skltn):
                min_id = min(mulperson_skltn.keys())
                print("prediced label is :", mulperson_label[min_id])

            img_displayer.display(img_disp, wait_key_ms=1)
            video_writer.write(img_disp)


            skels_to_save = get_the_skeleton_data_to_save_to_disk(mulperson_skltn)
            lib_commons.save_listlist(
                folder + skltn_folder_name +
                skltn_fname_format.format(ith_img),
                skels_to_save)
    finally:
        video_writer.stop()
        print("Program ends")
