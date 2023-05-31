import cv2
import numpy as np

# Initialize the image and colors
img_name = 'grid_pic.png'
GRID = 40

image = cv2.imread(img_name)
#Set different semi-transparent colors
COLORS = [(0, 0, 255, 128), (255, 0, 0, 128), (0, 255, 0, 128), (0, 255, 255, 128),
          (255, 0, 255, 128), (75, 0, 130, 128), (255, 165, 0, 128),(0, 255, 165, 128)]
COLOR_IND = 0

def draw_cube(event, x, y, flags, param):
    global image, COLOR_IND
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get mouse cell position
        grid_x = x // GRID
        grid_y = y // GRID

        # Get top left and bottom right coords
        start_x = grid_x * GRID
        start_y = grid_y * GRID
        end_x = start_x + GRID
        end_y = start_y + GRID

        # Draw a cube
        overlay = image.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), COLORS[COLOR_IND], -1, cv2.LINE_AA)
        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    # Scroll through colors
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            COLOR_IND = (COLOR_IND + 1) % len(COLORS)
        else:
            COLOR_IND = (COLOR_IND - 1) % len(COLORS)

# Create a window and set the mouse callback func
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_cube)

while True:
    cv2.imshow('Image', image)
    # Check  events
    key = cv2.waitKey(1) & 0xFF
    # Exit on 'q'
    if key == ord('q'):
        break
cv2.imwrite('modified_grid_pic.png', image)
cv2.destroyAllWindows()
