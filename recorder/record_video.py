import cv2

# Set the video codec and create a VideoWriter object
print('Setting everything up')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
output_filename = 'my_street.avi'

fps = 30.0              # Frames per second. We don't need 30fps
output_width = 640      # Output video width
output_height = 480     # Output video height

# Create a VideoWriter object to save the video
out = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))
cap = cv2.VideoCapture(1)  # Use 0 for the default camera. Mine is Xsplit app and 1 is the real one
record = False
print('Starting streaming')
while True:
    works, frame = cap.read()
    if not works:
        break
    # Write the frame to the output video file
    if record:
        out.write(frame)
    # Display the frame while recording
    cv2.imshow('My street', frame)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        if record:
            print('stop recording')
            record = False
        else:
            print('start recording')
            record = True

    # Break the loop if 'q' is pressed
    elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
print('Finished streaming')
cap.release()
out.release()
cv2.destroyAllWindows()
