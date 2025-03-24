#!/usr/bin/env python3
import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal

class Robot(Node):
    def __init__(self):
        super().__init__('robot')
        
        # Initialise a publisher to send movement commands to the robot base
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Flags for colour detection
        self.green_found = False
        self.red_found = False
        self.green_area = 0  # Store the area of the detected green contour
        
        # Sensitivity for HSV thresholds
        self.sensitivity = 10
        
        # Movement parameters (tune as necessary)
        self.forward_speed = 0.2
        self.backward_speed = -0.2
        
        # Define a target area for the green object (used to decide forward/backward movement)
        self.target_area = 2000
        
        # Initialise CvBridge and subscribe to the camera image topic
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription  # Prevent unused variable warning

    def callback(self, data):
        try:
            # Convert the ROS image to an OpenCV image
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: " + str(e))
            return
        
        # Display the camera feed
        cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', image)
        cv2.resizeWindow('camera_Feed', 320, 240)
        cv2.waitKey(3)
        
        # Convert image to HSV colour space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define HSV bounds for green
        green_lower = np.array([60 - self.sensitivity, 100, 100])
        green_upper = np.array([60 + self.sensitivity, 255, 255])
        green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
        
        # Define HSV bounds for red (using two ranges to cover the hue wrap-around)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([0 + self.sensitivity, 255, 255])
        red_lower2 = np.array([180 - self.sensitivity, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Process green mask: find contours and determine if a valid green object exists
        contours_green, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours_green:
            c_green = max(contours_green, key=cv2.contourArea)
            area_green = cv2.contourArea(c_green)
            if area_green > 500:  # Minimum area threshold for a valid detection
                self.green_found = True
                self.green_area = area_green
                # Draw a circle around the green object
                (gx, gy), gradius = cv2.minEnclosingCircle(c_green)
                cv2.circle(image, (int(gx), int(gy)), int(gradius), (0, 255, 0), 2)
            else:
                self.green_found = False
                self.green_area = 0
        else:
            self.green_found = False
            self.green_area = 0
        
        # Process red mask: find contours and determine if a valid red object exists
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours_red:
            c_red = max(contours_red, key=cv2.contourArea)
            area_red = cv2.contourArea(c_red)
            if area_red > 500:  # Minimum area threshold for a valid detection
                self.red_found = True
                # Draw a circle around the red object
                (rx, ry), rradius = cv2.minEnclosingCircle(c_red)
                cv2.circle(image, (int(rx), int(ry)), int(rradius), (0, 0, 255), 2)
            else:
                self.red_found = False
        else:
            self.red_found = False
        
        # Display the processed image with detected objects highlighted
        cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
        cv2.imshow('Processed', image)
        cv2.resizeWindow('Processed', 320, 240)
        cv2.waitKey(3)

    def walk_forward(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = self.forward_speed
        self.publisher.publish(desired_velocity)
        self.get_logger().info("Walking forward.")

    def walk_backward(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = self.backward_speed
        self.publisher.publish(desired_velocity)
        self.get_logger().info("Walking backward.")

    def stop(self):
        desired_velocity = Twist()  # All values zero (stopping the robot)
        self.publisher.publish(desired_velocity)
        self.get_logger().info("Stopping robot.")

def main():
    def signal_handler(sig, frame):
        robot.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    robot = Robot()
    
    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            # If red is detected, stop the robot immediately
            if robot.red_found:
                robot.stop()
            # Otherwise, if green is detected, decide movement based on its size
            elif robot.green_found:
                if robot.green_area < robot.target_area:
                    # Green object is small (far away) → move forward
                    robot.walk_forward()
                elif robot.green_area > robot.target_area:
                    # Green object is large (too close) → move backward
                    robot.walk_backward()
                else:
                    robot.stop()
            else:
                robot.stop()
            time.sleep(0.1)
    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
