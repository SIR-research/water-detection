"""
Script to verify all examples in the readme.
Simply execute
    python test_readme_examples.py
"""
from __future__ import print_function, division

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image
import json
import os
import copy
import shutil


def main():
    # old = r"/home/u33427/waterdetection/augmentation/dataset/train"
    # new = r"/home/u33427/waterdetection/augmentation/dataset/train/aug"
    
    # # Rotate 10 degrees
    # data_augmentation(old,
    #                   new, "rotate_10",
    #                   iaa.Sequential([iaa.Affine(rotate=10)]))
    # # # -10 degrees
    # data_augmentation(old,
    #                   new, "rotate_-10",
    #                   iaa.Sequential([iaa.Affine(rotate=-10)]))
      # # Rotate 25 degrees
    # data_augmentation(old,
    #                   new, "rotate_25",
    #                   iaa.Sequential([iaa.Affine(rotate=25)]))
    # # # -25 degrees
    # data_augmentation(old,
    #                   new, "rotate_-25",
    #                   iaa.Sequential([iaa.Affine(rotate=-25)]))
    
    
    
    # Gaussian Blur (three levels)
    # data_augmentation(old,
    #                   new, "GaussianBlur_low",
    #                   iaa.Sequential([iaa.GaussianBlur(sigma=1)]))
    # data_augmentation(old,
    #                   new, "GaussianBlur_mid",
    #                   iaa.Sequential([iaa.GaussianBlur(sigma=2)]))
    # data_augmentation(old,
    #                   new, "GaussianBlur_high",
    #                   iaa.Sequential([iaa.GaussianBlur(sigma=3)]))
    # # ( )
    # data_augmentation(old,
    #                   new, "AdditiveGaussianNoise_5",
    #                   iaa.Sequential([iaa.AdditiveGaussianNoise(scale=5)]))
    # data_augmentation(old,
    #                   new, "AdditiveGaussianNoise_10",
    #                   iaa.Sequential([iaa.AdditiveGaussianNoise(scale=10)]))
    # # Brightness change
    # data_augmentation(old,
    #                   new, "light_1.15",
    #                   iaa.Sequential([iaa.Multiply(mul=1.15)]))
    # data_augmentation(old,
    #                   new, "light_1.3",
    #                   iaa.Sequential([iaa.Multiply(mul=1.3)]))
    # data_augmentation(old,
    #                   new, "light_0.85",
    #                   iaa.Sequential([iaa.Multiply(mul=0.85)]))
    # data_augmentation(old,
    #                   new, "light_0.7",
    #                   iaa.Sequential([iaa.Multiply(mul=0.7)]))
    # # 
    # data_augmentation(old,
    #                   new, "Affine_scale_1.5",
    #                   iaa.Sequential([iaa.Affine(scale={"x": 1.5, "y": 1.5})]))
    # data_augmentation(old,
    #                   new, "Affine_scale_0.8",
    #                   iaa.Sequential([iaa.Affine(scale={"x": 0.8, "y": 0.8})]))



    # # data aug [XY, X, Y] (100,-100,20,-20)
    
    # data_augmentation(old,
    #                   new, "Affine_xy_20",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": 20, "y": 20})]))
    
    # data_augmentation(old,
    #                   new, "Affine_xy_-20",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": -20, "y": -20})]))
    
    # data_augmentation(old,
    #                   new, "Affine_xy_100",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": 100, "y": 100})]))
    
    # data_augmentation(old,
    #                   new, "Affine_xy_-100",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": -100, "y": -100})]))
    
    # data_augmentation(old,
    #                   new, "Affine_x_100",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": 100, "y": 0})]))
    
    # data_augmentation(old,
    #                   new, "Affine_x_-100",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": -100, "y": 0})]))
    
    # data_augmentation(old,
    #                   new, "Affine_y_-100",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": 0, "y": -100})]))
    
    # data_augmentation(old,
    #                   new, "Affine_y_100",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": 0, "y": 100})]))
    
    # data_augmentation(old,
    #                   new, "Affine_x_20",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": 20, "y": 0})]))
    
    # data_augmentation(old,
    #                   new, "Affine_x_-20",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": -20, "y": 0})]))
    
    # data_augmentation(old,
    #                   new, "Affine_y_-20",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": 0, "y": -20})]))
    
    # data_augmentation(old,
    #                   new, "Affine_y_20",
    #                   iaa.Sequential([iaa.Affine(translate_px={"x": 0, "y": 20})]))
    
    
    ##Data aug perspectiva
    
    # data_augmentation(old,
    #                   new, "perspective",
    #                   iaa.Sequential(iaa.PerspectiveTransform(scale=(0.01, 0.15))))
    
    
    # # # Data aug temperature
    
    # data_augmentation(old,
    #                   new, "temperature_4000K",
    #                   iaa.Sequential(iaa.ChangeColorTemperature(4000)))
    # data_augmentation(old,
    #                   new, "temperature_20000K",
    #                   iaa.Sequential(iaa.ChangeColorTemperature(20000)))
    
    
    # # Data aug grayscale
        
    # data_augmentation(old,
    #                   new, "grayscale_1",
    #                   iaa.Sequential(iaa.Grayscale(alpha=1.0)))
    
    
    # # Data aug flip
    
    data_augmentation(old,
                      new, "flip_horizontal",
                      iaa.Sequential([iaa.Fliplr(1.0)]))    
    ################################################################################ ##############################################################
    merge_data(old, new)
    ############################################################################### ###############################################################
	# # Mirror
    # flip_all("All files in the directory are mirrored")
    print("foi q foi kraio")

def flip_all(datasets_path):
    #All files in the source directory
    datasets_path_children = os.listdir(datasets_path)
    for temp in datasets_path_children:
        data_augmentation(os.path.join(datasets_path, temp),
                          os.path.join(datasets_path, temp), "_flip",
                          iaa.Sequential([iaa.Fliplr(1)]))


def merge_data(datasets_path, new_dataset):
    # Empty catalog collection
    datasets = []
    #All files in the source directory
    datasets_path_children = os.listdir(datasets_path)
    # 
    for datasets_path_child in datasets_path_children:
        tmp_path = os.path.join(datasets_path, datasets_path_child)
        if os.path.isdir(tmp_path):
            datasets.append(str(tmp_path))
    # Create a new directory
    if not os.path.exists(new_dataset):
        os.makedirs(new_dataset)
    ################################################################################################# ########################################################################
    # 
    annotations = {}
    # merge annotation
    for dataset in datasets:
        annotation = json.load(open(os.path.join(dataset, "via_region_data.json")))
        annotations.update(annotation)
    # Write the annotation file
    with open(os.path.join(new_dataset, "via_region_data.json"), 'w') as f:
        json.dump(annotations, f)
    ######################################################################################### ########################################################################
    for dataset in datasets:
        files = os.listdir(dataset)
        for file in files:
            # Copy only pictures
            if not file.endswith('json'):
                shutil.copyfile(os.path.join(dataset, file), os.path.join(new_dataset, file))


def data_augmentation(dataset_dir_old, dataset_dir_new_prefix, iaa_name, seq):
    print("Data Extension By flip: Executing!")
    #determine the law of transformation
    seq_det = seq.to_deterministic()
    # Determine if the folder exists, if not, create it
    dataset_dir_new = dataset_dir_new_prefix + iaa_name
    if os.path.exists(dataset_dir_new).__eq__(False):
        os.makedirs(dataset_dir_new)
    #Loading callout information
    annotations = json.load(open(os.path.join(dataset_dir_old, "via_region_data.json")))
    annotations_new = copy.deepcopy(annotations)
    annotations_new_keys = []
    #Get the key-value pair (old)
    for key in annotations_new:
        annotations_new_keys.append(key)

    # Don't have the outermost key, the inner layer is List
    annotations_values = list(annotations.values())
    # Determine if there is a Regions property and build a new List
    annotations_values = [a for a in annotations_values if a['regions']]

    #  
    for i, (annotations_value) in enumerate(annotations_values):
        # corresponding key points
        key_points_old = []
        if type(annotations_value['regions']) is dict:
            polygons = [r['shape_attributes'] for r in annotations_value['regions'].values()]
        else:
            polygons = [r['shape_attributes'] for r in annotations_value['regions']]
        # 
        filename = annotations_value['filename']
        image_old = Image.open(os.path.join(dataset_dir_old, filename))
        image_old = np.array(image_old)
        # polygons List, including a map of multiple Regions
        for j, (b) in enumerate(polygons):
            # Increase the key point of the picture
            key_points = []
            for k in range(0, len(b['all_points_x'])):
                try:
                    x_old = annotations_new[annotations_new_keys[i]]['regions'][j]['shape_attributes']['all_points_x'][
                        k]
                    y_old = annotations_new[annotations_new_keys[i]]['regions'][j]['shape_attributes']['all_points_y'][
                        k]
                    x = b['all_points_x'][k]
                    y = b['all_points_y'][k]
                    # print('old:(%d,%d) new(%d,%d)' % (x_old, y_old, x, y))
                    key_points.append(ia.Keypoint(x=x, y=y))
                except IndexError:
                    print("Error: i:" + str(i) + " name:" + annotations_new_keys[i] + " j:" + str(j) + " k:" + str(k))
            key_points_old.append(ia.KeypointsOnImage(key_points, shape=image_old.shape))
        # 
        image_new = seq_det.augment_image(image_old)
        #Keypoint transformation, is a List, multiple Region
        key_points_new = seq_det.augment_keypoints(key_points_old)

        # 
        image_file_name = filename.replace(".jpg", "_" + iaa_name + ".jpg")
        image_path_new = os.path.join(dataset_dir_new, image_file_name)
        #Save new image
        image_new = Image.fromarray(image_new.astype('uint8')).convert('RGB')
        image_new.save(image_path_new, "PNG")
        # Get the file size first
        image_size = os.path.getsize(image_path_new)
        # Replace Json's Key
        annotations_new.update({image_file_name + str(image_size): annotations_new.pop(annotations_new_keys[i])})
        #Updated Key
        annotations_new_keys[i] = image_file_name + str(image_size)
        #update filename
        annotations_new[annotations_new_keys[i]]['filename'] = image_file_name
        # update size
        annotations_new[annotations_new_keys[i]]['size'] = image_size

        # traverse the transformed point set (new), the same number as the old point, where idx is equivalent to j above
        for j in range(0, len(key_points_new)):
            for k, (key_point) in enumerate(key_points_new[j].keypoints):
                x_old = annotations_new[annotations_new_keys[i]]['regions'][j]['shape_attributes']['all_points_x'][k]
                y_old = annotations_new[annotations_new_keys[i]]['regions'][j]['shape_attributes']['all_points_y'][k]
                x_new = key_point.x
                y_new = key_point.y
                annotations_new[annotations_new_keys[i]]['regions'][j]['shape_attributes']['all_points_x'][k] = x_new
                annotations_new[annotations_new_keys[i]]['regions'][j]['shape_attributes']['all_points_y'][k] = y_new
        # # 
            image_old = key_points_old[j].draw_on_image(image_old)
        # # 
            image_new = key_points_new[j].draw_on_image(image_new)
        # #  
        ia.imshow(np.concatenate((image_old, image_new), axis=1))
    # print(type(annotations_new))  
    
    
      
    print('Data extension By flip: Done! ')

    with open(os.path.join(dataset_dir_new, "via_region_data.json"), 'w') as f:
        print(annotations_new)
        print(type(annotations_new))
        
        dumped = json.dumps(annotations_new, cls=NumpyEncoder)
        
        
        # dumped.replace(" ", "")
        # dumped = dumped[1:-1]
        
        # import ast        
        
        dumped = json.loads(dumped)
        
        json.dumps(dumped)
        
        
        print(dumped)
                
        json.dump(dumped, f)
        
    print('Data extension By flip: Done! ')
    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return int(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        # obj.replace("\\", "")
        return json.JSONEncoder.default(self, obj)
    


if __name__ == "__main__":
    main()

