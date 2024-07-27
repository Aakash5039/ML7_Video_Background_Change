# ML7_Video_Background_Change

A python script for video composting .

It takes two videos as input and produces a single video composite as output.The person in the first video is segmented out and blended seamlessly into the second video (background) .It uses deeplab v3 for segmentation and form harmonization , we use image harmonization and the link which we have loaded for image harmonization is https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2. Here we also used mediapy to read the videos .Since the tensorflow model was trained on PASCAL VOC dataset, we can segment out any object belonging to those classes and combine them with the other video to produce a video composite.

# **Requirements**

- Python
- TensorFlow
- OpenCV
- PIL
- MediaPy

# Prerequisites

Download model files : https://drive.google.com/open?id=1XRxpVu2I70Pu0IchXXEcfq9GDQw8XeT5
GPU with CUDA support

# Procedure

As per the requirement of the project, we started with reading two videos and converting them into frames. And then, giving the first image to the model , followed with normalization and with some preprocessing techniques(Guassian, Rotate). Then, finally suoerimposing the blurmap which is the outcome of the Guassian filters with the second video and combining the frames to make a file and saving it.
