#!/usr/bin/env python3
import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal

class ColourIdentifier(Node):
    def __init__(self):
        super().__init__('colour_identifier')
        # Initialise flag for green detection (default to False)
        self.green_found = False
        # Set sensitivity for the HSV threshold
        self.sensitivity = 10
        # Initialise CvBridge and subscribe to the image topic
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription  # prevent unused variable warning

    def callback(self, data):
        try:
            # Convert the ROS Image message to an OpenCV image
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: " + str(e))
            return
        
        # Display the original camera feed
        cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', image)
        cv2.resizeWindow('camera_Feed', 320, 240)
        cv2.waitKey(3)
        
        # Convert the image from BGR to HSV colour space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define HSV bounds for green
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        
        # Create a mask that only contains green regions
        mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        
        # Find contours in the mask (using RETR_LIST and CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour based on area
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            # Check if the area is large enough to be considered valid (adjust threshold as needed)
            if area > 500:
                self.green_found = True
                # Calculate the minimum enclosing circle to get the center and radius
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                self.get_logger().info("Green object detected with area: {:.2f}".format(area))
            else:
                self.green_found = False
        else:
            self.green_found = False

        # Display the result with the detected circle (if any)
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.imshow('Result', image)
        cv2.resizeWindow('Result', 320, 240)
        cv2.waitKey(3)

def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()

    rclpy.init(args=None)
    colour_identifier = ColourIdentifier()
    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(colour_identifier,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            time.sleep(0.1)
    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
