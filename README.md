# MAJOR
This repository consists of code for my final year project. We are aiming to build a mobile application that detects objects and estimates distance of the object from 
the camera using monocular depth estimation techniques.

depth_color_detection_webcam.py uses the Torch MiDas model to estimate depth.
keras_depth.py uses Keras to implement a U-Net architecture to train the model for depth estimation using the DIODE dataset. depth_estimator_webcam.py implements the model trained in keras_depth.py to fetch live feed from the camera and estimate depth on the same.
