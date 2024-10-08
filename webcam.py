import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

# Now import the Ollama module
import ollama

# Load the pre-trained cascade classifier for face detection
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Configure logging
log.basicConfig(filename='webcam.log', level=log.INFO)

# Initialize video capture from camera index 0 (change if needed)
video_capture = cv2.VideoCapture(0)
anterior = 0

# Initialize variables for image and response text display
display_image = None
display_text = ""

# Initialize face_filename with a placeholder value
face_filename = ""

# Function to run Ollama chat API with the image and insert new lines
def describe_image(image_path, max_line_length=35):
    res = ollama.chat(
        model="llava:7b",
        messages=[
            {
                'role': 'user',
                'content': 'Compliment features of my face with a short response:',
                'images': [image_path]
            }
        ]
    )
    content = res['message']['content']
    lines = []
    current_line = ''
    for word in content.split():
        if len(current_line) + len(word) + 1 <= max_line_length:
            current_line += f"{word} "
        else:
            lines.append(current_line.strip())
            current_line = f"{word} "
    if current_line:
        lines.append(current_line.strip())
    return '\n'.join(lines)

# Main loop for capturing and processing frames
while True:
    # Check if the camera is opened successfully
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Check if the spacebar is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Save the face ROI as an image
            face_filename = f"captured_face_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(face_filename, roi_color)
            log.info(f"Saved face image: {face_filename}")

            # Describe the saved face image using Ollama API
            display_text = describe_image(face_filename)
            display_image = roi_color

    # Display the image and text on the left side of the screen
    if display_image is not None:
        frame[0:display_image.shape[0], 0:display_image.shape[1]] = display_image

        lines = display_text.split('\n')  # Split display_text by newline character

        # Render each line of text separately
        y_offset = display_image.shape[0] + 30
        for line in lines:
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y_offset += 30  # Increase y offset for the next line

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Check for the 'q' key press to quit the program
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

