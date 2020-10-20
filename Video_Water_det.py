# -*- coding: utf-8 -*-
"""
Python code for testing the video on Mask R-CNN proposed by Girshick et al
and implimented by Matterport in Tensorflow and Keras.


@author: Nirav Raiyani
"""
# Importing the required Python packages
import os
import sys
import tensorflow as tf
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import cv2
import json_tricks as jsontrk

ROOT_DIR = os.getcwd()



sys.path.append(ROOT_DIR)  # Specifies the path for looking the following packages
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log
import balloon

# Creating the directory to save logs and weights of the model
MODEL_DIR = os.path.join(ROOT_DIR,"logs")

# Loading the configuration:Object name, No. of epochs and all hyperparameters
config = balloon.BalloonConfig() # Configurations are defined in 'balloon.py' and 'config.py'


# To modify (if needed) some setting in config.
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    

import cv2
import numpy as np


def random_colors(N):
    np.random.seed(42)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image

def make_video(outvid, images=None, fps=30, size=None,
               is_color=True, format="FMP4"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid



if __name__ == '__main__':
    """
        test everything
    """
 
    video_name = sys.argv[1]
    epoch = sys.argv[2]

    # We use a K80 GPU with 24GB memory, which can fit 3 images.
    batch_size = 1

    
    VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
    VIDEO_SAVE_DIR = os.path.join(VIDEO_DIR, "base_flip", video_name)
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "logs/water20200916T_flip_train/mask_rcnn_water_0"+epoch+".h5")
    #if not os.path.exists(COCO_MODEL_PATH):
    #    utils.download_trained_weights(COCO_MODEL_PATH)


    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    class_names = ['BG', 'Water']

    capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_name))
    print(os.path.join(VIDEO_DIR, video_name))

    try:
        if not os.path.exists(VIDEO_SAVE_DIR):
            os.makedirs(VIDEO_SAVE_DIR)
    except OSError:
        print ('Error: Creating directory of data')
    frames = []
    frame_count = 0
    # these 2 lines can be removed if you dont have a 1080p camera.
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        # Bail out when the video file ends
        if not ret:
            break
        
        # Save each frame of the video to a list
        skip_n_frames = 1
        frame_count += 1
        if frame_count % skip_n_frames == 0:
          
          frames.append(frame)
          print('frame_count :{0}'.format(frame_count))
          if len(frames) == batch_size:
              results = model.detect(frames, verbose=0)
              print('Predicted')
              for i, item in enumerate(zip(frames, results)):
                  frame = item[0]
                  r = item[1]

                  frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

                  name = '{0:04d}_{1}_{2}.jpg'.format(frame_count + i - batch_size,epoch,video_name.replace(".mp4",""))
                  name = os.path.join(VIDEO_SAVE_DIR, name)
                  cv2.imwrite(name, frame)
                  print('writing to file:{0}'.format(name))
                    
                  filename, _ = os.path.splitext(name)

                  ## save to numpy file
                  np.save(filename, r)
                  
                  ## save to json with json_tricks
                  with open(filename + '.json', 'w') as f:
                      f.write("#generated with json_tricks python package to encode numpy arrays.\n")
                      jsontrk.dump(r, f) 
                  
            # Clear the frames array to start the next batch
                  frames = []

    capture.release()
    make_video("out"+video_name,frames)
    
    
    
