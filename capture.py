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

    # Display the resulting frame
    cv2.imshow('Press Space to Capture', frame)

    # Wait for key press
    key = cv2.waitKey(1)

    # If space is pressed, save the image
    if key == 32:  # ASCII value of Spacebar
        image_path = os.path.join(directory, f"{image_counter}.png")
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")
        image_counter += 1
    elif key == 27:  # ASCII value of ESC
        break

# When everything done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
