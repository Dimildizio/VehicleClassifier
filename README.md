# VehicleClassifier

VehicleClassifier is a CNN project aimed at detecting and tracking vehicles and people in images and videos. It also includes functionality to count the traffic of large vehicles like trucks and buses (that cause damage to the road). 

## Goal

The primary goal of this project is to serve as a supplementary tool for a non-profit activity, aiming to lower traffic levels and reduce the cost of maintaining a specific private road, which, due to the code in this repo and some negotiation magic, it has already fulfilled since all the responsible actors and shareholders have come to an agreement to fund the project and install a barrier on the road to control traffic and prevent unauthorised vehicles from using said road.

## Project Structure

The project is organized into the following folders:

> img_manipulation: Contains code snippets for image manipulation tasks such as drawing a grid and bounding boxes.

> pictures: A folder that stores the input images to work on.

> recorder: Includes files for recording webcam videos using OpenCV (record_video.py) and capturing photos from the webcam stream on button press (take_pics.py).

> transfer_learning: A Google Colab notebook (train_model.ipynb) where transfer learning is performed to train a model for object detection.

> video_data: Contains raw videos and manipulated videos with detection boxes and counting lines.

### Additionally, there are a few other files:

> **Vehicle_count.ipynb**: This is the **main file** that performs object detection and tracking using YOLOv8 and ByteTracker. It also counts the number of passing vehicles.

> object_detection_ssdlite_mobilenet.py: Implements object detection using the ssdlite_mobilenet model on live video feed.

> yolov8_naive_application.py: Provides a basic application of YOLOv8 for object detection.

> LICENSE: The license file that specifies the permissions and restrictions for using the code.(usual MIT)

> .gitignore: A file that specifies which files and directories should be ignored by version control (e.g., files generated during runtime).

> requirements.txt: A list of requirements to use. Don't forget to install CUDA, CuDNN, onnx, TensorRT

## Getting Started

To get started with the VehicleClassifier project, you can follow these steps:

1. Install the necessary dependencies and libraries mentioned in the project.

2. Prepare the input data, including images and videos, in the appropriate folders.

3. Explore the code snippets in the img_manipulation folder to perform various image manipulation tasks.

4. Use the recorder module to record webcam videos or capture photos from the webcam stream.

5. Refer to the transfer_learning notebook to perform transfer learning and train a model for object detection.

6. Run the Vehicle_count.ipynb notebook for object detection, tracking, and vehicle counting.

7. Experiment with the object_detection_ssdlite_mobilenet.py script for real-time object detection using the ssdlite_mobilenet model.

8. Check the video_data folder for the processed videos with detection boxes and counting lines.

## Que vadis?

The project can be scaled for using via web app (mb Flask or Streamlit). I also got plans to implement it on mobile devices for custom objects tracking and counting.
