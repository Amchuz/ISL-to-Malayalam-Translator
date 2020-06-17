if True: 
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

import sys, os, time, argparse, logging
import cv2

sys.path.append(ROOT + "progs/githubs/tf-pose-estimation-master")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common

MAX_FRACTION_OF_GPU_TO_USE = 0.4
IS_DRAW_FPS = True

def _set_logger():
    logger = logging.getLogger('TfPoseEstimator')
    logger.setLevel(logging.DEBUG)
    logging_stream_handler = logging.StreamHandler()
    logging_stream_handler.setLevel(logging.DEBUG)
    logging_formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    logging_stream_handler.setFormatter(logging_formatter)
    logger.addHandler(logging_stream_handler)
    return logger

def _set_config(): 
    import tensorflow as tf
    from tensorflow import keras
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction=MAX_FRACTION_OF_GPU_TO_USE
    return config

def _get_input_img_size_from_string(image_size_str):
    width, height = map(int, image_size_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)


class SkeletonDetector(object):

    def __init__(self, model="cmu", image_size="432x368"):

        assert(model in ["mobilenet_thin", "cmu"])
        self._w, self._h = _get_input_img_size_from_string(image_size)
        
        self._model = model
        self._resize_out_ratio = 4.0 
        self._config = _set_config()
        self._tf_pose_estimator = TfPoseEstimator(get_graph_path(self._model), target_size=(self._w, self._h),tf_config=self._config)
        self._prev_t = time.time()
        self._cnt_image = 0
        
        self._logger = _set_logger()
        

    def detect(self, image):

        self._cnt_image += 1
        if self._cnt_image == 1:
            self._image_h = image.shape[0]
            self._image_w = image.shape[1]
            self._scale_h = 1.0 * self._image_h / self._image_w
        t = time.time()

        humans = self._tf_pose_estimator.inference(
            image, resize_to_default=(self._w > 0 and self._h > 0),
            upsample_size=self._resize_out_ratio)


        elapsed = time.time() - t
        self._logger.info('inference image in %.4f seconds.' % (elapsed))

        return humans
    
    def draw(self, img_disp, humans):
  
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)
        if IS_DRAW_FPS:
            cv2.putText(img_disp,
                        "fps = {:.1f}".format( (1.0 / (time.time() - self._prev_t) )),
                        (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        self._prev_t = time.time()

    def humans_to_skels_list(self, humans, scale_h = None): 

        if scale_h is None:
            scale_h = self._scale_h
        skeletons = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(18*2)
            for i, body_part in human.body_parts.items(): 
                idx = body_part.part_idx
                skeleton[2*idx]=body_part.x
                skeleton[2*idx+1]=body_part.y * scale_h
            skeletons.append(skeleton)
        return skeletons, scale_h
    

def test_openpose_on_webcamera():
    
    from lib.io import ReadFromWebcam, ImageDisplayer
    webcam_reader = ReadFromWebcam(max_framerate=10)
    img_displayer = ImageDisplayer()
       
    skeleton_detector = SkeletonDetector("mobilenet_thin", "432x368")

    import itertools
    for i in itertools.count():
        img = webcam_reader.read_image()
        if img is None:
            break
        print(f"Read {i}th image...")

        humans = skeleton_detector.detect(img)
        
        img_disp = img.copy()
        skeleton_detector.draw(img_disp, humans)
        img_displayer.display(img_disp)
        
    print("Program ends")

if __name__ == "__main__":
    test_openpose_on_webcamera()
