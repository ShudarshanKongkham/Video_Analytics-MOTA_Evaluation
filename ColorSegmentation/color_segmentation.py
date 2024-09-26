import cv2
import numpy as np

# Fixed HSV values for segmentation
hue_min, hue_max = 23, 69
sat_min, sat_max = 80, 255
val_min, val_max = 107, 255

# Initialize video capture
cap = cv2.VideoCapture("video_data/yellowFish.mp4")  # Use the video file

# List to store the path history of centroids
paths = []

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        # Reinitialize video capture if the end of video is reached
        cap = cv2.VideoCapture("video_data/yellowFish.mp4")  # Reinitialize video
        ret, frame = cap.read()
        if not ret:
            break

    # Resize frame
    frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the HSV values
    lower_bound = np.array([hue_min, sat_min, val_min])
    upper_bound = np.array([hue_max, sat_max, val_max])

    # Create a mask using the HSV range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the frame to draw on
    output_frame = frame.copy()

    # Initialize new centroids list
    new_centroids = []

    # Draw bounding boxes around detected objects and track centroid
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green bounding box

            # Calculate the centroid of the bounding box
            cx = x + w // 2
            cy = y + h // 2
            new_centroids.append((cx, cy))

            # Draw centroid
            cv2.circle(output_frame, (cx, cy), 5, (0, 255, 0), -1)  # Red circle for centroid

    # Update the path history with new centroids
    if len(new_centroids) > 0:
        # Append new centroids to paths
        for centroid in new_centroids:
            # Check if the path for this centroid already exists
            found = False
            for path in paths:
                if len(path) > 0 and np.linalg.norm(np.array(path[-1]) - np.array(centroid)) < 50:
                    path.append(centroid)
                    found = True
                    break
            if not found:
                paths.append([centroid])

    # Keep only the last 10 points in each path
    for path in paths:
        if len(path) > 25:
            del path[:-25]

    # Draw paths
    for path in paths:
        if len(path) > 1:
            for i in range(len(path) - 1):
                cv2.line(output_frame, path[i], path[i + 1], color=(0, 0, 255), thickness=2)

    # Debug: Show mask to ensure proper segmentation
    cv2.imshow('Mask', mask)
    
    # Display the frame with bounding boxes, centroids, and path
    cv2.imshow('Segmented with Bounding Boxes and Path Tracking', output_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
