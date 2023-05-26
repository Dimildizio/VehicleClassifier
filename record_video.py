import cv2

# Set the video codec and create a VideoWriter object
print('Setting everything up')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_filename = 'my_street.avi'

fps = 15.0              # Frames per second. We don't need 30fps
output_width = 640      # Output video width
output_height = 480     # Output video height

# Create a VideoWriter object to save the video
out = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))
cap = cv2.VideoCapture(1)  # Use 0 for the default camera. Mine is Xsplit app and 1 is the real one

print('Starting recording')
while True:
    works, frame = cap.read()
    if not works:
        break
    # Write the frame to the output video file
    out.write(frame)
    # Display the frame while recording
    cv2.imshow('My street', frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('Finished recording')
cap.release()
out.release()
cv2.destroyAllWindows()