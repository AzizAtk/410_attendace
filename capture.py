import cv2
import os

# Ask for the student's name
student_name = input("Enter the student's name: ")

# Create a directory for the student if it doesn't exist
directory = f"./{student_name}"
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Counter for the image file names
image_counter = 1

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame capture was successful, process the image
    if ret:
        # Display the original frame in color
        cv2.imshow('Press Space to Capture', frame)

        # Convert to Grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Histogram Equalization on the original colored image
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        channels = list(cv2.split(ycrcb))  # Convert tuple to list for modification
        channels[0] = cv2.equalizeHist(channels[0])
        equalized_frame = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)
        
        # Apply Gaussian Blur to the original colored image to reduce noise
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
        
        # Resize the original image (optional, specify desired dimensions as needed)
        resized_frame = cv2.resize(frame, (256, 256))  # Example: resize to 256x256 pixels

    # Wait for key press
    key = cv2.waitKey(1)

    # If space is pressed, save the images
    if key == 32:  # ASCII value of Spacebar
        image_path = os.path.join(directory, f"{image_counter}.png")
        # Save the original frame
        cv2.imwrite(image_path, frame)
        # Save the processed frames
        cv2.imwrite(image_path.replace(".png", "_gray.png"), gray_frame)
        cv2.imwrite(image_path.replace(".png", "_equalized.png"), equalized_frame)
        cv2.imwrite(image_path.replace(".png", "_blurred.png"), blurred_frame)
        cv2.imwrite(image_path.replace(".png", "_resized.png"), resized_frame)
        print(f"Original and processed images saved: {image_path}")
        image_counter += 1
    elif key == 27:  # ASCII value of ESC
        break

# When everything done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
