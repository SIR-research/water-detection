#%%
 
#%%
import os
import numpy as np 

import matplotlib.pyplot as plt
import cv2
import sys

#%%

def get_metadata(path):
    '''
    Return the irrigation metadata dictionary containing a vector of 
    image 'data' (Region of Interest 'roi', 'score', and Water 'mask'), 
    the water 'areas' detected, the 'mean' and 'std'.
    
    Parameters:
        argument1 (str): Water Detection images path
        
    Returns:
        dict: metadata
    '''
     
    img_data=[]
    
    for filename in os.listdir(path):
        if filename.endswith(".npy"):
            
            # if np.random.rand()>0.5:    #temp: remove half of frames
            #     continue
            
            print(filename)
                
            r = np.load(path+'/'+filename, allow_pickle=True)
                  
            
            img_data.append({'roi': r.item()['rois'],
                         'score':r.item()['scores'],
                         'mask':r.item()['masks']
                        })
            
            
            # plt.imshow(img)
            # plt.show()
          
            
            continue
        else:
            continue
    
    #creates the metadata
    
    areas=np.array([])
    for d in img_data:
        img = d['mask'][:,:,0]
        areas = np.append(areas, np.sum(img))


    irrmap=0
    for d in img_data:
        
        img = d['mask'][:,:,0]
        irrmap += img 
                    
    
    metadata = {'data': img_data,
                'irrmap': irrmap,
                'areas': areas,
                'mean': np.mean(areas),
                'std': np.std(areas),
                'video_url': path}

    return metadata

#%%

def debug_rescale(img, scale=0.5):

    
    imgnorm = img*255/np.max(img)
    imgres = cv2.resize(imgnorm, (round(imgnorm.shape[1]*scale), round(imgnorm.shape[0]*scale)))

   
    
    if scale<1:
        
        B = np.zeros(img.shape)
        A = imgres
        
        nb = B.shape
        na = A.shape
        lowerx = (nb[0]) // 2 - (na[0] // 2)
        lowery = (nb[1]) // 2 - (na[1] // 2)
        
        upperx = (nb[0] // 2) + (na[0] // 2)
        uppery = (nb[1] // 2) + (na[1] // 2)
        B[lowerx:upperx, lowery:uppery] = A
        
        imgdone = B
    
    else:
        
                        
        y = (imgres.shape[0]-img.shape[0]) // 2
        h = img.shape[0]
        x = (imgres.shape[1]-img.shape[1]) // 2
        w = img.shape[1]
        
        
        imgdone = imgres[y:y+h, x:x+w]

        


    
    return imgdone
    


def debug_get_metadata_rescale(path, scale=0.5):
    '''
    Return the irrigation metadata dictionary containing a vector of 
    image 'data' (Region of Interest 'roi', 'score', and Water 'mask'), 
    the water 'areas' detected, the 'mean' and 'std'.
    
    Parameters:
        argument1 (str): Water Detection images path
        
    Returns:
        dict: metadata
    '''    

     
    img_data=[]
    
    for filename in os.listdir(path):
        if filename.endswith(".npy"):
            
            # if np.random.rand()>0.1:    #temp: remove half of frames
            #     continue
            
            print(filename)
                
            r = np.load(path+'/'+filename, allow_pickle=True)
                  
            
            img_data.append({'roi': r.item()['rois'],
                         'score':r.item()['scores'],
                         'mask':r.item()['masks']
                        })
            
            
            # plt.imshow(img)
            # plt.show()
          
            
            continue
        else:
            continue
    
    # debug resize images
    for d in img_data:
        d['mask'] = debug_rescale(d['mask'][:,:,0], scale=scale)
        d['mask'] = np.expand_dims(d['mask'], axis=2)
        d['mask'] = d['mask']>0

    
    #creates the metadata
    
    areas=np.array([])
    for d in img_data:
        img = d['mask'][:,:,0]
        areas = np.append(areas, np.sum(img))


    irrmap=0
    for d in img_data:
        
        img = d['mask'][:,:,0]
        irrmap += img 
                    
    
    metadata = {'data': img_data,
                'irrmap': irrmap,
                'areas': areas,
                'mean': np.mean(areas),
                'std': np.std(areas),
                'video_url': path
                }

    return metadata
    


    
#%%


def create_bar_plot(gtmd, vermd, comp, comparison_path):
        
    plt.figure(200)
    plt.tight_layout()
    
    
    height = [gtmd['mean'], vermd['mean']]
    bars = ('GR', 'Verification')
    y_pos = np.arange(len(bars))
    
    error= [gtmd['std'], vermd['std']]
        
    if comp['comparison_result']['value'] == 'OK' :
        second_color = 'green'
    elif comp['comparison_result']['value'] == 'NOK' :
        second_color = 'red'
        
    color = ['blue', second_color]
    
    plt.bar(y_pos, height, yerr=error, color=color, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks(y_pos, bars)
    plt.ylabel('Irrigation Area (pixels)')
    plt.title('Irrigation Average Coverage Area')


    plt.savefig(comparison_path+'/bar_plot.png')
    
    return {"value": comparison_path + '/bar_plot.png', "type": "url"}
    
    # plt.show()
           
                    

def draw_bars_comparison(gtmd, vermd, second_color, comparison_path):

    plt.figure(200)
    plt.tight_layout()
    
    
    height = [gtmd['mean'], vermd['mean']]
    bars = ('GR', 'Verification')
    y_pos = np.arange(len(bars))
    
    error= [gtmd['std'], vermd['std']]
        
    color = ['blue', second_color]
    
    plt.bar(y_pos, height, yerr=error, color=color, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks(y_pos, bars)
    plt.ylabel('Irrigation Area (pixels)')
    plt.title('Irrigation Average Coverage Area')


    plt.savefig(comparison_path+'/bars_plot_comparison.png', format='png', transparent=False)
    
    # plt.show()


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask != 0,
            image[:, :, n] * (1 - alpha*mask) + alpha*mask * c,
            image[:, :, n]
        )
    return image




# def draw_comparison(gtmd, vermd):

    
#     # normalize vector
#     gtimg = gtmd['irrmap']   
    
#     # img = np.round(img*255/np.max(img)) #from 0 to 255
#     gtimg = gtimg/np.max(gtimg) #from 0 to 1
    
    
    
#     # join matricess Blue for the gtmd and Red for the vermd
#     image = np.ones(gtimg.shape+(3,)) #creates new image array of dimention [x,y,3]
    
    
#     verimg = vermd['irrmap']
#     verimg = verimg/np.max(verimg)
    

   
    
#     image = apply_mask(image,gtimg,(0,0,1), alpha=0.5)
    
#     if vermd['mean'] < (gtmd['mean']-gtmd['std']) or vermd['mean'] > (gtmd['mean']+gtmd['std']):
       
#        image = apply_mask(image,verimg,(1,0,0), alpha=0.5)
    
       
#        draw_bars_comparison(gtmd, vermd, 'red')
    
#     else:
#         image = apply_mask(image,verimg,(0,1,0), alpha=0.5)
#         draw_bars_comparison(gtmd, vermd, 'green')
    
    
      
    
#     imgmsk = np.logical_or(gtimg>0, verimg>0)
    
#     ### crop image
#     thresh = imgmsk*255
    
#     cnts = cv2.findContours(thresh, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     c = max(cnts, key=cv2.contourArea)
    
#     # Obtain outer coordinates
#     left = tuple(c[c[:, :, 0].argmin()][0])
#     right = tuple(c[c[:, :, 0].argmax()][0])
#     top = tuple(c[c[:, :, 1].argmin()][0])
#     bottom = tuple(c[c[:, :, 1].argmax()][0])
    
    
#     print(bottom, top, left, right)
#     margin=0.1
#     y_margin = np.int((bottom[1] - top[1])*margin)
#     x_margin = np.int((right[0] - left[0])*margin)
#     print(x_margin, y_margin)
    
#     image = image[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]
    
    
#     ## display image
    
    
    
    
#     print(np.unique(gtimg))
#     print(np.unique(image))
    
#     # image = 1-image    
    
#     #display images

    

#     plt.figure(300)
#     # imgplot = plt.imshow(gtimg, cmap='Blues', alpha=1, origin='lower')
#     # imgplot = plt.imshow(verimg, cmap='Reds', alpha=0.5, origin='lower')
#     plt.imshow(image)


#     # plt.show()

#     return imgmsk



#%%
    

def plot_comparison(gtmd, vermd, comparison_path):

        
    # normalize vector
    gtimg = gtmd['irrmap']   
    
    # img = np.round(img*255/np.max(img)) #from 0 to 255
    gtimg = gtimg/np.max(gtimg) #from 0 to 1
    
    
    verimg = vermd['irrmap']
    verimg = verimg/np.max(verimg)
    
    
    # join matricess Blue for the gtmd and Red for the vermd
    image = np.ones(gtimg.shape+(3,)) #creates new image array of dimention [x,y,3]
    imagegt = np.ones(gtimg.shape+(3,)) 
    imagever = np.ones(gtimg.shape+(3,)) 
    
    
    imagegt = apply_mask(imagegt,gtimg,(0,0,1), alpha=0.5)
    image = apply_mask(image,gtimg,(0,0,1), alpha=0.5)
    
    if vermd['mean'] < (gtmd['mean']-gtmd['std']) or vermd['mean'] > (gtmd['mean']+gtmd['std']):
       
       image = apply_mask(image,verimg,(1,0,0), alpha=0.5)
       imagever = apply_mask(imagever,verimg,(1,0,0), alpha=0.5)
    
       draw_bars_comparison(gtmd, vermd, 'red', comparison_path)
    
    else:
        image = apply_mask(image,verimg,(0,1,0), alpha=0.5)
        imagever = apply_mask(imagever,verimg,(0,1,0), alpha=0.5)
                
        draw_bars_comparison(gtmd, vermd, 'green', comparison_path)
    
    
    imgmsk = np.logical_or(gtimg>0, verimg>0)
    
    ### crop image
    thresh = imgmsk*255
    
    cnts = cv2.findContours(thresh, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    
    # Obtain outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    
    
    margin=0.1
    y_margin = np.int((bottom[1] - top[1])*margin)
    x_margin = np.int((right[0] - left[0])*margin)
    
    image = image[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]

    imagegt = imagegt[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]
    imagever = imagever[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]
    
    
    #display images

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    

    fig.tight_layout()
    
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
    ax1.set_xlabel('Groud Truth')        
    ax2.set_xlabel('Verification')        
    ax3.set_xlabel('Overlay')        
    
    plt.suptitle('Irrigation Area Comparison', x=0.5, y=0.85)
    
    ax1.imshow(imagegt)
    ax2.imshow(imagever)
    ax3.imshow(image)
    
    
    plt.savefig(comparison_path+'/irrigation_comparison.png', format='png', transparent=False)

    # plt.show()

#%%

def create_area_plot(gtmd, vermd, comp, comparison_path):

        
    # normalize vector
    gtimg = gtmd['irrmap']   
    
    # img = np.round(img*255/np.max(img)) #from 0 to 255
    gtimg = gtimg/np.max(gtimg) #from 0 to 1
    
    
    verimg = vermd['irrmap']
    verimg = verimg/np.max(verimg)
    
    
    # join matricess Blue for the gtmd and Red for the vermd
    image = np.ones(gtimg.shape+(3,)) #creates new image array of dimention [x,y,3]
    imagegt = np.ones(gtimg.shape+(3,)) 
    imagever = np.ones(gtimg.shape+(3,)) 
    
    
    imagegt = apply_mask(imagegt,gtimg,(0,0,1), alpha=0.5)
    image = apply_mask(image,gtimg,(0,0,1), alpha=0.5)
    
    if comp['comparison_result']['value'] == 'OK' :
        image = apply_mask(image,verimg,(0,1,0), alpha=0.5)
        imagever = apply_mask(imagever,verimg,(0,1,0), alpha=0.5)
                
    
    elif comp['comparison_result']['value'] == 'NOK' :
    
       image = apply_mask(image,verimg,(1,0,0), alpha=0.5)
       imagever = apply_mask(imagever,verimg,(1,0,0), alpha=0.5)
    
    
    imgmsk = np.logical_or(gtimg>0, verimg>0)
    
    ### crop image
    thresh = imgmsk*255
    
    cnts = cv2.findContours(thresh, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv2.contourArea)
    
    # Obtain outer coordinates
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bottom = tuple(c[c[:, :, 1].argmax()][0])
    
    
    margin=0.1
    y_margin = np.int((bottom[1] - top[1])*margin)
    x_margin = np.int((right[0] - left[0])*margin)
    
    image = image[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]

    imagegt = imagegt[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]
    imagever = imagever[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]
    
    
    #display images

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    

    fig.tight_layout()
    
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
    ax1.set_xlabel('Groud Truth')        
    ax2.set_xlabel('Verification')        
    ax3.set_xlabel('Overlay')        
    
    plt.suptitle('Irrigation Area Comparison', x=0.5, y=0.85)
    
    ax1.imshow(imagegt)
    ax2.imshow(imagever)
    ax3.imshow(image)
    
    
    plt.savefig(comparison_path+'/area_plot.png')

    return {"value": comparison_path + '/area_plot.png', "type": "url"}

    # plt.show()




#%%
import json

def create_comparison_entity_json(gtmd, vermd, id="comparison"):
    


    comparison_ngsi = {
        "id": id,
        "type": "irrigation_comparison",
        "ground_truth": {
            "type": "StruturedValue",
            "value": { 
                "mean": gtmd['mean'],
                "std": gtmd['std'],
                "video_url": gtmd['video_url']
                }
        },
        "verification": {
            "type": "StruturedValue",
            "value": {
                "mean": vermd['mean'],
                "std": vermd['std'],
                "video_url": vermd['video_url']
                }
        }
    }
        # "bar_plot": {
        #     "value": comparison_path + '/bar_plot.png',
        #     "type": "url"
        # },
        # "area_plot": {
        #     "value": comparison_path + '/area_plot.png',
        #     "type": "url"
        # }
        
    return comparison_ngsi
    
    
def save_comparison_json(comparison_ngsi, comparison_path):
            
    with open(comparison_path + '/comparison_ngsi.json', 'w') as fp:
        json.dump(comparison_ngsi, fp, indent=4)




def get_comparison_result(gtmd, vermd):
    
    if vermd['mean'] < (gtmd['mean']-gtmd['std']) or vermd['mean'] > (gtmd['mean']+gtmd['std']):
        return {'value': 'NOK', 'type': 'Text'}
    
    else:
        return {'value': 'OK','type': 'Text'}


#%%

import requests

def create_orion_entity(entity):
    
    # url_entities = 'http://localhost:1026/v2/entities'
#    url_entities = 'http://177.104.61.52:1026/v2/entities' #SWAMP PRODUCTION IP!!!
    url_entities = 'http://177.104.61.47:1026/v2/entities' #swamp test url
    
    url_subscription = 'http://localhost:1026/v2/subscriptions'
    headers = {'content-type': 'application/json'}

    r = requests.post(url_entities, data=json.dumps(entity), headers=headers)
    
    print(r.text)


def compare(GT_VIDEO_NAME, VER_VIDEO_NAME, save_entity=False):

    # ROOT_DIR = os.getcwd()
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

   
    GT_DIR = os.path.join(ROOT_DIR, "videos/base_flip", GT_VIDEO_NAME)
    VER_DIR = os.path.join(ROOT_DIR, "videos/base_flip", VER_VIDEO_NAME)
    
    
    # gt_video_name_no_ext = os.path.splitext(GT_VIDEO_NAME)[0]
    # ver_video_name_no_ext = os.path.splitext(VER_VIDEO_NAME)[0]

    comparison_name = GT_VIDEO_NAME + '_comp_' + VER_VIDEO_NAME

    COMPARISON_PATH = ROOT_DIR + '/comparison/' + comparison_name

    print(COMPARISON_PATH)
    if not os.path.exists(COMPARISON_PATH):
        os.makedirs(COMPARISON_PATH)
        print('criou')
    

    gt_metadata = get_metadata(GT_DIR)
    ver_metadata = get_metadata(VER_DIR)
    # ver_metadata = debug_get_metadata_rescale(VER_DIR, scale=0.5)
    
    comparison_entity = create_comparison_entity_json(gt_metadata, ver_metadata, id=comparison_name)
    
    comparison_entity['comparison_result'] = get_comparison_result(gt_metadata, ver_metadata)
    
    
            # "bar_plot": {
            #     "value": comparison_path + '/bar_plot.png',
            #     "type": "url"
            # },
            # "area_plot": {
            #     "value": comparison_path + '/area_plot.png',
            #     "type": "url"
            # }
    
    comparison_entity['bar_plot'] = create_bar_plot(gt_metadata, ver_metadata, comparison_entity, COMPARISON_PATH)
    
    comparison_entity['area_plot'] = create_area_plot(gt_metadata, ver_metadata, comparison_entity, COMPARISON_PATH)
    
    
    save_comparison_json(comparison_entity, COMPARISON_PATH)

    if save_entity:
        create_orion_entity(comparison_entity)

## ver entidades no navegador
# http://177.104.61.47:1026/v2/entities/
# http://177.104.61.47:1026/v2/entities/comparison_1
# http://177.104.61.47:1026/v2/entities/?type=irrigation_comparison

#%%


# ROOT_DIR = os.getcwd()

# GT_VIDEO_NAME = 'GT.mp4'
# VER_VIDEO_NAME = 'VER.mp4'

# GT_DIR = os.path.join(ROOT_DIR, "water-detection/videos/base_flip", GT_VIDEO_NAME)
# VER_DIR = os.path.join(ROOT_DIR, "water-detection/videos/base_flip", VER_VIDEO_NAME)

# COMPARISON_PATH = ROOT_DIR + '/comparison/' + GT_VIDEO_NAME + '_comp_' + VER_VIDEO_NAME

# print(COMPARISON_PATH)
# if not os.path.exists(COMPARISON_PATH):
#     os.makedirs(COMPARISON_PATH)
#     print('criou')




# print(GT_DIR)
# print(VER_DIR)
# print(type(VER_DIR))

# # detect_water(VER_VIDEO_NAME, 100, skip_n_frames=1)


# gt_metadata = get_metadata(GT_DIR)
# ver_metadata = debug_get_metadata_rescale(VER_DIR, scale=0.5)

# comparison_entity = create_comparison_entity_json(gt_metadata, ver_metadata)

# comparison_entity['comparison_result'] = get_comparison_result(gt_metadata, ver_metadata)


#         # "bar_plot": {
#         #     "value": comparison_path + '/bar_plot.png',
#         #     "type": "url"
#         # },
#         # "area_plot": {
#         #     "value": comparison_path + '/area_plot.png',
#         #     "type": "url"
#         # }

# comparison_entity['bar_plot'] = create_bar_plot(gt_metadata, ver_metadata, comparison_entity, COMPARISON_PATH)

# comparison_entity['area_plot'] = create_area_plot(gt_metadata, ver_metadata, comparison_entity, COMPARISON_PATH)


# save_comparison_json(comparison_entity, COMPARISON_PATH)

# create_orion_entity(comparison_entity)

#%%

# from water_detection import detect_water


if __name__ == '__main__':
    
    
    ROOT_DIR = os.getcwd()

    GT_VIDEO_NAME = sys.argv[1]
    VER_VIDEO_NAME = sys.argv[2]
    
    GT_DIR = os.path.join(ROOT_DIR, "videos/base_flip", GT_VIDEO_NAME)
    VER_DIR = os.path.join(ROOT_DIR, "videos/base_flip", VER_VIDEO_NAME)
    
    COMPARISON_PATH = ROOT_DIR + '/comparison/' + GT_VIDEO_NAME + '_comp_' + VER_VIDEO_NAME
    
    
    print(COMPARISON_PATH)
    if not os.path.exists(COMPARISON_PATH):
        os.makedirs(COMPARISON_PATH)
        print('criou')
    
    
    
    
    print(GT_DIR)
    print(VER_DIR)
    print(type(VER_DIR))
    
    # detect_water(VER_VIDEO_NAME, 100, skip_n_frames=1)
    
    
    gt_metadata = get_metadata(GT_DIR)
    ver_metadata = debug_get_metadata_rescale(VER_DIR, scale=0.5)
    
    comparison_entity = create_comparison_entity_json(gt_metadata, ver_metadata)
    
    comparison_entity['comparison_result'] = get_comparison_result(gt_metadata, ver_metadata)
    
    
            # "bar_plot": {
            #     "value": comparison_path + '/bar_plot.png',
            #     "type": "url"
            # },
            # "area_plot": {
            #     "value": comparison_path + '/area_plot.png',
            #     "type": "url"
            # }
    
    comparison_entity['bar_plot'] = create_bar_plot(gt_metadata, ver_metadata, comparison_entity, COMPARISON_PATH)
    
    comparison_entity['area_plot'] = create_area_plot(gt_metadata, ver_metadata, comparison_entity, COMPARISON_PATH)
    
    
    save_comparison_json(comparison_entity, COMPARISON_PATH)

    create_orion_entity(comparison_entity)