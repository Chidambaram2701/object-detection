import cv2

# Load the Haar Cascade classifier for face detection
# Make sure the XML file is in the same directory as this script
face_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Start video capture from the default webcam
# The argument '0' refers to the first webcam found
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Loop to continuously capture frames
while True:
    # Read a single frame from the video capture
    # 'ret' is a boolean that is True if the frame was read correctly
    # 'frame' is the image data
    ret, frame = cap.read()

    # If the frame was not read correctly, break the loop
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale, as Haar Cascades work best on grayscale images
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    # The 'detectMultiScale' function returns a list of rectangles (x, y, w, h)
    # The parameters scaleFactor and minNeighbors can be tuned for better detection
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30) # Minimum size of the object to be detected
    )

    # Loop through the detected faces and draw a rectangle around each one
    for (x, y, w, h) in faces:
        # Draw the rectangle on the original color frame
        # (x, y) is the top-left corner, (x+w, y+h) is the bottom-right
        # (0, 255, 0) is the color of the rectangle in BGR format (Green)
        # 2 is the thickness of the line
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame with the rectangles
    cv2.imshow('Live Object Recognition', frame)

    # Wait for a key press. If 'q' is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop, release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()