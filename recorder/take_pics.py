import cv2

# Set the video codec and create a VideoWriter object
print('Setting everything up')
output_width = 640      # Output video width
output_height = 480     # Output video height

def write_photo(frame, n):
    num = n+1
    cv2.imwrite(f'train_pic_{num}.png', frame)
    print(f'img #{num} created')
    return num
# Create a VideoWriter object to save the video
#out = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))
cap = cv2.VideoCapture(1)  # Use 0 for the default camera. Mine is Xsplit app and 1 is the real one

print('Starting streaming')
n=38
while True:
    works, frame = cap.read()
    if not works:
        break
    # Write the frame to the output video file
    # Display the frame while recording
    cv2.imshow('My street', frame)
    if cv2.waitKey(1) & 0xFF == ord('w'):
       n =  write_photo(frame, n)
    # Break the loop if 'q' is pressed
    elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
print('Finished streaming')
cap.release()

cv2.destroyAllWindows()
