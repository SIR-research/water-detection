import streamlit as st
import cv2 as cv 
import tempfile 




st.image('assets/swamp_logo.png', width=100)



st.title('SWAMP Irrigation Comparison Service')



st.write("This service detects irrigation failure comparing two videos.")


option = st.selectbox(
    '',
     ['Register Ground Truth',
     'Compare Irrigation'])


if option == 'Register Ground Truth':
  st.write("""

    Upload a Ground Truth video.

    The Ground True video is a footage of the irrigation system  working properly.

    This video will be used for future comparative analysis.""")

  f_gt = st.file_uploader("Select the Ground Truth video.") 
  

  if f_gt:

    tfile = tempfile.NamedTemporaryFile(delete=False) 

    tfile.write(f_gt.read()) 

    video_gt = cv.VideoCapture(tfile.name)

    #display video
    image_placeholder_gt = st.empty()
    while True:
        success, image = video_gt.read()

        image_placeholder_gt.image(image, width=400)


    pressed = st.button('Register Ground Truth')
    if pressed:
        st.write("""chamar função para fazer inferência!
                    apresentar vídeo com água detectada
                    perguntar se confirma groundtruth e então salvar metadata""")




elif option == 'Compare Irrigation':

  st.write("""

    Upload a video to be compared to Ground Truth video.

    This video must be captured in similar weather and equipment conditions as the Ground Truth video.""")

  f_comp = st.file_uploader("Select the video to be compared to the Ground Truth.") 

  if f_comp:

    tfile = tempfile.NamedTemporaryFile(delete=False) 

    tfile.write(f_comp.read()) 

    video_comparison = cv.VideoCapture(tfile.name)


    #display video
    image_placeholder_comp = st.empty()
    while True:
        success, image = video_comparison.read()

        image_placeholder_comp.image(image, width=400)



    pressed = st.button('Compare Irrigation')
    if pressed:
        st.write("""chamar função para fazer inferência!
                    rodar algoritmo comparativo
                    plotar imagens de comparação.""")




