import cv2
import torch
from ultralytics import YOLO

    
def model_inference(model_run, one_frame):
    result = model_run(one_frame)
    return result[0].plot()


# Define a video to work on
VIDEO_LOCATION = "video_data/raw_video/output_street.avi"
OUTPUT_LOCATION = "video_data/processed_video/street_detected.avi"

# Create a model to perform object detection
MODEL = "yolov8s.pt"  # there are s(mall),m(edium),l(arge) and x(tended) models that vary in speed and accuracy
model = YOLO(MODEL)
model.fuse()

# Put the model on gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('current torch version is:', torch.__version__)
print('Running device is:', device)
model.to(device)

# Get the video to work on, create encoder and a VideoWriter object to save the video
cap = cv2.VideoCapture(VIDEO_LOCATION)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
size = (int(cap.get(3)), int(cap.get(4)))  # width and height of original file
fps = 10
output_video = cv2.VideoWriter(OUTPUT_LOCATION, fourcc, fps, size)

# Check how many total frames in our original video
FRAME_NUM_TO_USE = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Number of frames', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

for num in range(FRAME_NUM_TO_USE):
    # Show current frame number
    print(f'current frame: {num} of {FRAME_NUM_TO_USE}')
    works, frame = cap.read()
    if not works:
        break
    # apply the model, write and show the model
    infer = model_inference(model, frame)
    output_video.write(infer)
    cv2.imshow('Myframe', infer)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()

'''
Summary:
So if we perform the code on cpu it takes quite a while on inference. In my case 1 frame per 1.5sec. Which is not acceptable.
The solution of course is to use GPU with CUDA and optimize the model in pattern YOLO->ONYX->TensorRT
For ease of use most of the code should be written in google colab
The only reason to use any kind of IDE - to test it on local cpu/gpu and showing video is much easier compared to colab as well
For the current task Object tracking is required (likely supervision, bytetrack). Also naive application of Kalman filter could be written for practice.
For object counting supervision will likely be used (LineCounter) - for trucks and cars that cross a line. 
'''
