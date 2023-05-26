import cv2
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
#from imutils.video import FPS, VideoStream
#from imutils.object_detection import non_max_suppression
#from imutils.video import FPS


# Define the class labels. python 3.8 doesn't support torchvision with coco. the list is not full. there are more than 90
class_labels = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
                12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog',
                18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra',
                24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase',
                30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite',
                35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard',
                39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife',
                45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange',
                51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake',
                57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table',
                62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard',
                68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink',
                73: 'refrigerator', 74: 'book',
                75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier',
                80: 'toothbrush'} #and so on

#but we need only a few so we'll redefine the dict
class_labels = {1:'person', 2:'bicycle', 3:'car', 4:'motocycle', 6:'bus', 8:'truck'}

# Load the pre-trained model
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

# Set up the transformation pipeline
transform = transforms.Compose([
             transforms.ToTensor()])

def perform_object_detection(frame):#, tracker):
    # Preprocess the frame
    input_tensor = transform(frame).unsqueeze(0)

    # Run the frame through the model
    with torch.no_grad():
        predictions = model(input_tensor)

    # Extract relevant information from the predictions
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Apply confidence threshold
    confidence_threshold = 0.5

    filtered_boxes = [box for box, score, label in zip(boxes, scores, labels) if score > confidence_threshold]
    filtered_scores = [score for score, label in zip(scores, labels) if score > confidence_threshold]
    filtered_labels = [label for score, label in zip(scores, labels) if score > confidence_threshold]

    # If there are any objects detected
    if len(filtered_boxes) > 0:
        filtered_boxes = torch.stack(filtered_boxes)
        filtered_scores = torch.tensor(filtered_scores)
        # Apply non-maximum suppression 
        keep = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
        filtered_boxes = [filtered_boxes[i] for i in keep]
        filtered_labels = [filtered_labels[i] for i in keep]
    else:
        filtered_boxes, filtered_scores = [],[]

    #Apply tracker. Maybe it's better to use YOLO and supervision solutions for that
    #objects = tracker.update(filtered_boxes)
    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        #for (object_id,controid) in object.items():                           #with tracker
        xmin, ymin, xmax, ymax = box
        #xmin,ymin,xmax,ymax=  centroid                                        #with tracker
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        try:
            #avoid key error for detecting things not in the class labels list
            label_text = f"{class_labels[label.item()]}: {score:.2f}"
        except KeyError:
            continue
        #label_text = f'{class_labels[object_id]} {object_id}'                 #with tracker

        # Write the label name at the bottom right of the rectangle
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(frame, label_text, (int(xmin), int(ymax) + label_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.putText(frame, label_text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)             #with tracker
    return frame

#Create a tracker instance
#tracker = cv2.legacy.TrackerCSRT_create()
#fps = FPS()

#Set video device number - my device 0 is Xsplit app so device 1 is the real camera
video_device = 1
cap = cv2.VideoCapture(video_device)

while cap.isOpened():
    ret, frame = cap.read()
    #For tracker
    '''if not ret:
        break
    frame = perform_object_detection(frame,tracker)
    cv2.imshow('MyStreet', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    fps.update()'''
    
    #with no tracker
    if ret:
        frame = perform_object_detection(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        break

#fps.stop()
#print(f"FPS {fps.fps()}")
cap.release()
cv2.destroyAllWindows()
