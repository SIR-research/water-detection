# Water Detection Service

This service aims to detect malfunctioning irrigation systems using footage acquired by a drone.
A machine learning-based algorithm detects the water spray using a Mask Regional Convolutional Neural Network (MRCNN)¹. The service is available through a Streamlit web app.

1 - C. K. G. Albuquerque, S. Polimante, A. Torre-Neto and R. C. Prati, "Water spray detection for smart irrigation systems with Mask R-CNN and UAV footage," 2020 IEEE International Workshop on Metrology for Agriculture and Forestry (MetroAgriFor), Trento, 2020, pp. 236-240, doi: 10.1109/MetroAgriFor50201.2020.9277542. [See full article](https://ieeexplore.ieee.org/abstract/document/9277542)

## Installation

Switch to the service branch.
```bash
git checkout service
```

Create the Conda Virtual Environment using the file water_serv.yml located in the env/ directory. (linux-64 platforms)

```bash
conda env create -f env/water_serv.yml
```

Activate the new environment

```bash
conda activate water_serv
```

## Usage

Start the Streamlit web app.
```bash
streamlit run streamlit_app/app.py
```

The web app should launch as a new browser tab. If the web app is not launched, access it at the address specified on the terminal. The default address is http://localhost:8501.
