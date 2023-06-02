import cv2
from torchvision import models
from torchvision.transforms import functional as f

# Get the picture
img_name = 'raw_pic.png'
img = cv2.imread(img_name)

# Load the pre-trained model, let's use resnet
model = models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
model.eval()
print('Loaded model')
# Convert to pytorch tensor
tensor = f.to_tensor(img)
tensor = tensor.unsqueeze(0)

# Evaluate the tensor
result = model(tensor)

# Get ALL the bounding boxes. we don't need threshold
boxes = result[0]['boxes']

for box in boxes:
    x1, y1, x2, y2 = [int(num) for num in box.tolist()]
    cv2.rectangle(img, (x1, y1, x2, y2), (10, 10, 10), 2)

cv2.imshow('Boxes', img)
cv2.imwrite('show_detect_boxes.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
