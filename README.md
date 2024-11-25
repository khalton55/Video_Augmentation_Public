#Video Augmentation Project

Example video streaming and augmentation application using OpenCV and LLAVA on NVIDIA Jetson 

Supports Webcam stream, Video File, RTSP stream

Supports Grayscale, Text overlay, and Llava querying

Requirements: NVIDIA Jetson with jetson-containers installed Installed nano_llm:r36.3.0

To Run: 
Client - Edit config to include IP of machine running llava server, edit camera address (Windows: 0, Linux "/dev/video0" usually)
python3 video_aug_server.py 
(Most likely will have to install dependencies, will be fixed when able to make container)

Server - 

Run this command while in repo directory jetson-containers run -v .:/app --workdir /app --entrypoint /bin/bash dustynv/nano_llm:r36.3.0

python3 video_aug_server.py

TODO:

Containerize once jetson-container build bug is fixed Add support for flask on phone

Lower processing time
