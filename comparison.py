#%%

import os
import numpy as np 
import matplotlib.pyplot as plt


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
            
            if np.random.rand()>0.1:    #temp: remove half of frames
                continue
            
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
                'std': np.std(areas)
                }

    return metadata
#%%
# def draw_irrigation_map(md):
#     irrmap=md['irrmap']
            
         
#     plt.imshow(irrmap, cmap='Blues')
#     plt.show()
        

#%%
import cv2

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
            
            if np.random.rand()>0.1:    #temp: remove half of frames
                continue
            
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
                'std': np.std(areas)
                }

    return metadata
    

#%%
# calculates the irrigation map, mean and std deviation.        


gt_metadata = get_metadata(path = '/home/sergio/water-detection/videos/base_flip/irr_ok_srt.mp4')

# draw_irrigation_map(gt_metadata)



# ver_metadata = get_metadata(path = '/home/sergio/water-detection/videos/base_flip/irr_ok_srt.mp4')
ver_metadata = debug_get_metadata_rescale(path = '/home/sergio/water-detection/videos/base_flip/irr_ok_srt.mp4',
                                          scale=0.5)


# draw_irrigation_map(ver_metadata)

    
#%%
def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask != 0,
            image[:, :, n] * (1 - alpha*mask) + alpha*mask * c,
            image[:, :, n]
        )
    return image

def draw_bars_comparison(gtmd, vermd, second_color):

    plt.figure(200)
    height = [gtmd['mean'], vermd['mean']]
    bars = ('GR', 'Verification')
    y_pos = np.arange(len(bars))
    
    error= [gtmd['std'], vermd['std']]
        
    color = ['blue', second_color]
    
    plt.bar(y_pos, height, yerr=error, color=color, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks(y_pos, bars)
    plt.show()




def draw_comparison(gtmd, vermd):

    
    # normalize vector
    gtimg = gtmd['irrmap']   
    
    # img = np.round(img*255/np.max(img)) #from 0 to 255
    gtimg = gtimg/np.max(gtimg) #from 0 to 1
    
    
    
    # join matricess Blue for the gtmd and Red for the vermd
    image = np.ones(gtimg.shape+(3,)) #creates new image array of dimention [x,y,3]
    
    
    verimg = vermd['irrmap']
    verimg = verimg/np.max(verimg)
    

   
    
    image = apply_mask(image,gtimg,(0,0,1), alpha=0.5)
    
    if vermd['mean'] < (gtmd['mean']-gtmd['std']) or vermd['mean'] > (gtmd['mean']+gtmd['std']):
       
       image = apply_mask(image,verimg,(1,0,0), alpha=0.5)
    
       
       draw_bars_comparison(gtmd, vermd, 'red')
    
    else:
        image = apply_mask(image,verimg,(0,1,0), alpha=0.5)
        draw_bars_comparison(gtmd, vermd, 'green')
    
    
    
    
    
    
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
    
    
    print(bottom, top, left, right)
    margin=0.1
    y_margin = np.int((bottom[1] - top[1])*margin)
    x_margin = np.int((right[0] - left[0])*margin)
    print(x_margin, y_margin)
    
    image = image[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]
    
    
    ## display image
    
    
    
    
    print(np.unique(gtimg))
    print(np.unique(image))
    
    # image = 1-image    
    
    #display images

    

    plt.figure(300)
    # imgplot = plt.imshow(gtimg, cmap='Blues', alpha=1, origin='lower')
    # imgplot = plt.imshow(verimg, cmap='Reds', alpha=0.5, origin='lower')
    plt.imshow(image)
    plt.show()

    return imgmsk

msk = draw_comparison(gt_metadata, ver_metadata)


#%%


def draw_comparison_side_by_side(gtmd, vermd):

    
    # normalize vector
    gtimg = gtmd['irrmap']   
    
    # img = np.round(img*255/np.max(img)) #from 0 to 255
    gtimg = gtimg/np.max(gtimg) #from 0 to 1
    
    
    verimg = vermd['irrmap']
    verimg = verimg/np.max(verimg)
    
    
    # join matricess Blue for the gtmd and Red for the vermd
    image = np.ones(gtimg.shape+(3,)) #creates new image array of dimention [x,y,3]
    imagegt = np.ones(gtimg.shape+(3,)) #creates new image array of dimention [x,y,3]
    imagever = np.ones(gtimg.shape+(3,)) #creates new image array of dimention [x,y,3]
    
    

   
    
    imagegt = apply_mask(imagegt,gtimg,(0,0,1), alpha=0.5)
    image = apply_mask(image,gtimg,(0,0,1), alpha=0.5)
    
    if vermd['mean'] < (gtmd['mean']-gtmd['std']) or vermd['mean'] > (gtmd['mean']+gtmd['std']):
       
       image = apply_mask(image,verimg,(1,0,0), alpha=0.5)
       imagever = apply_mask(imagever,verimg,(1,0,0), alpha=0.5)
    
       draw_bars_comparison(gtmd, vermd, 'red')
    
    else:
        image = apply_mask(image,verimg,(0,1,0), alpha=0.5)
        imagever = apply_mask(imagever,verimg,(0,1,0), alpha=0.5)
                
        draw_bars_comparison(gtmd, vermd, 'green')
    
    
    
    
    
    
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
    
    
    print(bottom, top, left, right)
    margin=0.1
    y_margin = np.int((bottom[1] - top[1])*margin)
    x_margin = np.int((right[0] - left[0])*margin)
    print(x_margin, y_margin)
    
    image = image[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]

    imagegt = imagegt[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]
    imagever = imagever[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]
    
    
    #display images

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    for ax in fig.axes:
        ax.set_xticks([], [])
        ax.set_yticks([],[])
        
    ax1.set_xlabel('Groud Truth')        
    ax2.set_xlabel('Verification')        
    ax3.set_xlabel('Overlay')        
    
    # imgplot = plt.imshow(gtimg, cmap='Blues', alpha=1, origin='lower')
    # imgplot = plt.imshow(verimg, cmap='Reds', alpha=0.5, origin='lower')
    
    ax1.imshow(imagegt)
    ax2.imshow(imagever)
    ax3.imshow(image)
    

    plt.show()

    

draw_comparison_side_by_side(gt_metadata, ver_metadata)




#%%


thresh = msk*255


cnts = cv2.findContours(thresh, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if len(cnts) == 2 else cnts[1]
c = max(cnts, key=cv2.contourArea)

# Obtain outer coordinates
left = tuple(c[c[:, :, 0].argmin()][0])
right = tuple(c[c[:, :, 0].argmax()][0])
top = tuple(c[c[:, :, 1].argmin()][0])
bottom = tuple(c[c[:, :, 1].argmax()][0])


print(bottom, top, left, right)
margin=0.1
y_margin = np.int((bottom[1] - top[1])*margin)
x_margin = np.int((right[0] - left[0])*margin)
print(x_margin, y_margin)

crop_img = msk[top[1]-y_margin:bottom[1]+y_margin, left[0]-x_margin:right[0]+x_margin]
plt.imshow(crop_img)



