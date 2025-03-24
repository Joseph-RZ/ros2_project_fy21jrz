#!/usr/bin/env python3
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal

class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')
        # Initialize sensitivity (make sure to define it!)
        self.sensitivity = 20
        
        # Initialise CvBridge and set up subscriber to the image topic
        self.bridge = CvBridge()
        # Use the same topic as in first_step (adjust if needed)
        self.subscription = self.create_subscription(Image, 'camera/image_raw', self.callback, 10)
        self.subscription  # prevent unused variable warning

    def callback(self, data):
        try:
            # Convert the received image into an OpenCV image
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: " + str(e))
            return

        cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', image)
        cv2.resizeWindow('camera_Feed', 320, 240)
        cv2.waitKey(3)
        
        # Set the upper and lower bounds for green using self.sensitivity
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        
        # Convert the BGR image into an HSV image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # use hsv_image (all lowercase) for consistency

        # Define red bounds (two ranges) and blue bounds using self.sensitivity
        hsv_red_lower1 = np.array([0, 100, 100])
        hsv_red_upper1 = np.array([0 + self.sensitivity, 255, 255])
        hsv_red_lower2 = np.array([180 - self.sensitivity, 100, 100])
        hsv_red_upper2 = np.array([180, 255, 255])
        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])
        
        # Create masks for each colour using cv2.inRange()
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        red_mask1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)

        # Combine the masks for green, red, and blue
        combined_mask = cv2.bitwise_or(green_mask, red_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

        # Apply the combined mask to the original image
        result_image = cv2.bitwise_and(image, image, mask=combined_mask)

        # Optional: Extract contours from the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour based on area
            c = max(contours, key=cv2.contourArea)
            # Draw a bounding rectangle around the largest contour
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw the minimum enclosing circle around the largest contour
            ((cx, cy), radius) = cv2.minEnclosingCircle(c)
            cv2.circle(result_image, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)

        # Show the processed result
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Result', 320, 240)
        cv2.imshow('Result', result_image)
        cv2.waitKey(1)

def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()

    rclpy.init(args=None)
    cI = colourIdentifier()
    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
