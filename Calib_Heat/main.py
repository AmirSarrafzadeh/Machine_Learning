folder_path = r"M:\Python\Telegram\Calib_Heat\images"

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io


for dir, _, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.tiff'):
            file_path = os.path.join(dir, file)
            # Load the TIFF image
            image = io.imread(file_path)

            # Manually adjust the threshold value
            threshold_value = 34250
            _, binary_image = cv2.threshold(image, threshold_value, 65535, cv2.THRESH_BINARY)

            # Convert the binary image to 8-bit
            binary_image_8bit = (binary_image / 256).astype('uint8')

            # Apply morphological operations to enhance the contours
            kernel = np.ones((5, 5), np.uint8)
            morphed_image = cv2.morphologyEx(binary_image_8bit, cv2.MORPH_CLOSE, kernel)

            # Apply edge detection using Canny
            edges = cv2.Canny(morphed_image, 50, 150)

            # Find contours in the edge-detected image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort the contours by area (in descending order) to identify the largest two circles
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

            # Draw contours on the original image for visualization
            contour_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

            # Display the contours
            plt.figure(figsize=(10, 10))
            plt.imshow(contour_image)
            plt.title('Contours of the Circles (Enhanced)')
            plt.axis('off')
            plt.savefig(f'{file.split('.')[0]}_contours.png')
            # plt.show()

            # Get the bounding boxes of the two largest contours
            bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
            right_circle_bbox = bounding_boxes[0]
            left_circle_bbox = bounding_boxes[1]

            # Extract the circles from the original image using the bounding boxes
            right_circle = image[right_circle_bbox[1]:right_circle_bbox[1] + right_circle_bbox[3], right_circle_bbox[0]:right_circle_bbox[0] + right_circle_bbox[2]]
            left_circle = image[left_circle_bbox[1]:left_circle_bbox[1] + left_circle_bbox[3], left_circle_bbox[0]:left_circle_bbox[0] + left_circle_bbox[2]]

            # Resize the left circle to match the size of the right circle
            resized_left_circle = cv2.resize(left_circle, (right_circle.shape[1], right_circle.shape[0]))

            # Calculate the intensity of both circles
            right_circle_intensity = np.mean(right_circle)
            resized_left_circle_intensity = np.mean(resized_left_circle)

            print(f'Right Circle Intensity: {right_circle_intensity}')
            print(f'Resized Left Circle Intensity: {resized_left_circle_intensity}')




