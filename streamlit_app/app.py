import streamlit as st
import cv2
import tempfile 
import numpy as np
import importlib
import json

import time

# importlib.import_module('test')




import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../water_detection.py
import water_detection as wd

import comparison_v1 as comp
#import test


#gambiarra to keep state of session
import SessionState


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


session_state = SessionState.get(gt_name='')  # Pick some initial values.



def save_video(tempvideofile, path):
    import tempfile, shutil

    f = tempvideofile
    file_name = f.name
    f.close()
    shutil.copy(file_name, path)
    os.remove(file_name)




def read_video_place_preview(st_video):
    tfile = tempfile.NamedTemporaryFile(delete=False) 

    tfile.write(st_video.read()) 

    video_gt = cv2.VideoCapture(tfile.name)

    

    #display video (but holds code execution)

    # image_placeholder_gt = st.empty()
    # while True:
    #     success, image = video_gt.read()

    #     image_placeholder_gt.image(image, width=400)

    #display only first frame for reference
    image_placeholder_gt = st.empty()
    
    success, image = video_gt.read()

    image_placeholder_gt.image(image, width=400)

    return tfile


def display_n_images(st_video, n_img):
        st.markdown("""
        ----------------------  
        ## Water Detection Preview
        
        Examples of water detected on some frames throughout the video.
        """)        
        #display 10 images for reference.
        images = []
        metadata_dir = ROOT_PATH+"/videos/base_flip/"+st_video.name

        for file in os.listdir(metadata_dir):
            if file.endswith(".jpg"):
                images.append(os.path.join(metadata_dir, file))    
        n_img = 10
        img_idx = np.linspace(0,len(images)-1, n_img).astype(int)
        images = [images[i] for i in img_idx]
        st.image(images, width=300, caption=img_idx)


def compare_videos(gt_name, ver_name):
    st.markdown("""
        -----------------------
        ## Comparison Results
        
        """)      

    comp.compare(gt_name, ver_name, save_entity=False)


    comparison_name = gt_name + '_comp_' + ver_name
    
    st.markdown("""
        ### Area Plot Comparison

        Comparison of areas containing water.

        This plot displays the area of detected water on each frame on the Reference 
        video (left), verification video (middle), and the overlapping
        area of both.
        
        """) 
    st.image(ROOT_PATH+'/comparison/'+comparison_name+'/area_plot.png')

    st.markdown("""
        ### Average Irrigation

        Mean and Standard Deviation of areas containing water.

        This bar plot displays the average number of pixels detected as water on
        the Reference video (left bar) and the verification video (right bar). The
        error bars are the standard deviation.
        
        """) 
    st.image(ROOT_PATH+'/comparison/'+comparison_name+'/bar_plot.png')

    
    with open(ROOT_PATH+'/comparison/'+comparison_name+'/comparison_ngsi.json') as f:
        jdata = json.load(f)



    st.markdown("""
        ----------------------
        ## Final Result
        
        """) 

    if (jdata['comparison_result']['value'] == 'OK'):
        st.success('IRRIGATION OK')

    else:
        if (jdata["verification"]["value"]["mean"] < jdata["ground_truth"]["value"]["mean"]):
            st.error('UNDER IRRIGATION')
        else:
            st.error('OVER IRRIGATION')

    st.markdown("""
        --------

        ## Technical Information
        
        ### SWAMP Entity

        SWAMP Entity data referring to this comparison.

        
        """) 
    st.json(jdata)






import base64

LOGO_IMAGE = 'assets/swamp_logo.png'


st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:40px !important;
        color: #1d8ecd !important;
        padding-top: 40px !important;
    }
    .logo-img {
        float:right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" width="whatever" height=200 src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
        <p class="logo-text">SWAMP <br>Water Detection Service</p>
    </div>
    """,
    unsafe_allow_html=True
)




st.markdown("""
    
    This service detects failures on irrigation systems based on
    video footage.

    A video of the irrigation system working properly is called Reference. 
    
    * Upload a video of the system working properly at the side bar
     using the option **Register Reference video**.

    Under and over irrigation can be detected by in future 
    videos by comparing it to the Reference video.

    * Upload the irrigation system videos to compare it to the
     Reference video using the option **Compare Irrigation** on the side bar.

    ------------------------------------------------

    """)

# st.beta_expander("""C. K. G. Albuquerque, S. Polimante, A. Torre-Neto and R. C. Prati, "Water spray detection for smart irrigation systems with Mask R-CNN and UAV footage," 2020 IEEE International Workshop on Metrology for Agriculture and Forestry (MetroAgriFor), Trento, 2020, pp. 236-240, doi: 10.1109/MetroAgriFor50201.2020.9277542.""")

my_expander = st.beta_expander('Reference')
# my_expander.write()
my_expander.markdown('C. K. G. Albuquerque, S. Polimante, A. Torre-Neto and R. C. Prati, "Water spray detection for smart irrigation systems with Mask R-CNN and UAV footage," 2020 IEEE International Workshop on Metrology for Agriculture and Forestry (MetroAgriFor), Trento, 2020, pp. 236-240, doi: 10.1109/MetroAgriFor50201.2020.9277542.')
my_expander.markdown('''[Find the article](https://ieeexplore.ieee.org/document/9277542)''')


option = st.sidebar.selectbox(
    'Select the Service',
     ['Compare Irrigation',
     'Register Reference'])


if option == 'Register Reference':

  st.markdown("""
    # Reference Video Registration

    Select a video of the irrigatio system working properly.

    """)

  


#  session_state = SessionState.get(gt=0)  # Pick some initial values.
  f_gt = st.file_uploader("Upload Reference video") 
  




  if f_gt:

    st.markdown("""
        ## Video Preview
        """)
    # tfile = tempfile.NamedTemporaryFile(delete=False) 

    # tfile.write(f_gt.read()) 

    # video_gt = cv2.VideoCapture(tfile.name)

    

    # #display video (but holds code execution)

    # # image_placeholder_gt = st.empty()
    # # while True:
    # #     success, image = video_gt.read()

    # #     image_placeholder_gt.image(image, width=400)

    # #display only first frame for reference
    # image_placeholder_gt = st.empty()
    
    # success, image = video_gt.read()

    # image_placeholder_gt.image(image, width=400)

    session_state.gt = f_gt.name


    tfile = read_video_place_preview(f_gt)
    
    st.warning('Detecting water on video can take several minutes.')
    start_det_btn = st.button('Start Water Detection')
    if start_det_btn:

        #saves the video and remove temporary file.
        with st.spinner('Saving video file...'):
            save_video(tfile, ROOT_PATH+'/videos/'+f_gt.name)
        st.success('Video Saved.')
        
        #Detecting Water UNCOMMENT!
        with st.spinner('Detecting water on video frames...'):
            wd.detect_water(f_gt.name, 100, skip_n_frames=60)
        st.success('Finished Detecting Water!')

#        with st.spinner("""Detecting water on video frames...
#            this can take a while."""):    
#            time.sleep(3)
#        st.success('Finished Detecting Water!')

        display_n_images(f_gt, 10)


elif option == 'Compare Irrigation':

  st.markdown("""
    # Compare Irrigation

    Select a video of the irrigatio system to be compared to 
    Reference video.

    """)

  

  f_ver = st.file_uploader("Upload video for comparison") 

  if f_ver:
    st.markdown("""
        ## Video Preview
        """)

    tfile = read_video_place_preview(f_ver)

    st.warning('Detecting water on video can take several minutes.')
    start_comp_btn = st.button('Start Comparison!')
    if start_comp_btn:

        #saves the video and remove temporary file.
        with st.spinner('Saving video file...'):
            save_video(tfile, ROOT_PATH+'/videos/'+f_ver.name)
        st.success('Video Saved.')

         #Detecting Water UNCOMMENT!!
        with st.spinner('Detecting water on video frames...'):
            wd.detect_water(f_ver.name, 100, skip_n_frames=60)
        st.success('Finished Detecting Water!')

#        with st.spinner("""Detecting water on video frames...
#            this can take a while."""):    
#            time.sleep(3)
#        st.success('Finished Detecting Water!')

        display_n_images(f_ver, 10)


        compare_videos(session_state.gt, f_ver.name)









