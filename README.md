# VehicleClassifier

VehicleClassifier is a CNN project aimed at detecting and tracking vehicles and people in images and videos. It also includes functionality to count the traffic of large vehicles like trucks and buses (that cause damage to the road). 

## Goal

The primary goal of this project is to serve as a supplementary tool for a non-profit activity, aiming to lower traffic levels and reduce the cost of maintaining a specific private road, which, due to the code in this repo and some negotiation magic, it has been already fulfilled since all the responsible actors and shareholders have come to an agreement to fund the project and install a barrier on the road to control traffic and prevent unauthorised vehicles from using said road.

This is how the road looked before the project started.

![road_before](https://github.com/Dimildizio/VehicleClassifier/assets/42382713/f0e649c6-de37-4bfe-bc9d-92b7bdab0420)

## Project Structure

The project is organized into the following folders:

> img_manipulation: Contains code snippets for image manipulation tasks such as drawing a grid and bounding boxes.

> pictures: A folder that stores the input images to work on.

> recorder: Includes files for recording webcam videos using OpenCV (record_video.py) and capturing photos from the webcam stream on button press (take_pics.py).

> transfer_learning: A Google Colab notebook (train_model.ipynb) where transfer learning is performed to train a model for object detection.

> video_data: Contains raw videos and manipulated videos with detection boxes and counting lines.

### Model structure

In the prject YOLO v8 is used as a model with different variants (tiny, small, medium, extra large). Also other models like ssd mobilenet (for size and performance were tested). 


![YOLO_order](https://github.com/Dimildizio/VehicleClassifier/assets/42382713/e316b226-d267-4fc4-9fff-8885f6c7b97d)

The whole process should look according to the following pattern

![structure](https://github.com/Dimildizio/VehicleClassifier/assets/42382713/e6c39c6b-dfc6-4d41-a056-703d0977a000)

The **observations** were carried out from two different observation points (from about 40-60 feet and 240-260 feet with the former beeing more preferable since the smaller the object is the worse it gets detected). Smaller detenction models don't really do the job if placed high above.

![model_comparison](https://github.com/Dimildizio/VehicleClassifier/assets/42382713/9ce49267-1961-4dba-8f79-91b0747821c6)

> **ByteTracker** has been chosen as tracking model due to it's balance between accuracy and performance (image source: https://www.arxiv-vanity.com/papers/2110.06864)

![image](https://github.com/Dimildizio/VehicleClassifier/assets/42382713/1f8fe1f8-d6c7-4a2e-ad71-7da5bd5e0cbd)


> **Transfer learning** isn't really required since detecting truchs, buses and cars is a pretty common task. But still has been inplemented with metrics such as mAP being utilized.

![metrics_transfer](https://github.com/Dimildizio/VehicleClassifier/assets/42382713/a7b9b569-6fff-48c0-bad4-b597c92752ab)

> As **labeling** instrument Roboflow has been used with it's awesome labeling software that allowed to label data using segmentation in basically one click.

![segmentation_labeling](https://github.com/Dimildizio/VehicleClassifier/assets/42382713/4c7ee1c2-19d4-480e-a95d-7b80443b771e)

> We want to have a model that can detect, track and count objects that cross a certain line. Since we got in\out traffic we need two lines and the **result** looks like on the following picture.

![result_counting](https://github.com/Dimildizio/VehicleClassifier/assets/42382713/22105b99-51a2-4772-bf13-49c54f2cd4f1)



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

## Project effect

So far the project helped convince the majority of the shareholders in the necessity of installing a barrier. It takes a while but as an intermediate result on the road to the boom barrier remotely controlled from mobile device and\or radio keys, road barrier blocks have been installed. 
The disadvantege is that unlike the boom barrier, the blocks completely block (duh) the road instead of preventing uncontrolled usage of it, although the gate (on the picture) still provides access to the other side of the road and (which is important) only to those who own a remote key from it.
After all the required actions and reconciliations done, the blocks will be replaced by the boom barrier. 

![road_in_process](https://github.com/Dimildizio/VehicleClassifier/assets/42382713/f533561c-4a64-4078-8b07-244e9f216899)

Also on the brighter side, the problem has attracted lots of attention and I was able to force and speed up road works aimed to repare the roadbed. 
