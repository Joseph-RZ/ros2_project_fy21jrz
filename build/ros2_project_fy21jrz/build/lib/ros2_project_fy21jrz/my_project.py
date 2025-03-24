#!/usr/bin/env python3

import rclpy
import time
import math
import cv2
import numpy as np
import signal

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.exceptions import ROSInterruptException

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge, CvBridgeError

# Send a nav2goal message to the robot and when blue is detected along the way, enter into ORIENTING mode (until blue is centered), and then enter into APPROACHING mode (until 1m from blue box)
class BlueApproachNode(Node):

    def __init__(self):
        super().__init__('blue_approach_node')

        # State of the machine, which can be NAVIGATING, ORIENTING, APPROACHING, and GOAL_REACHED.
        # Currently we set it to NAVIGATING
        self.mode = "NAVIGATING"

        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_handle = None
        self.currently_navigating = False

        # Initiate the camera as shown in the lab
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Parameters:
        # Tolerance for the ORIENTATING mode to center the blue box.
        self.center_tolerance = 5
        self.rotate_speed = 0.3
        self.forward_speed = 0.2
        # Exit ORIENTATING mode after 20 seconds if blue is not detected.
        self.orientation_timeout = 20.0
        self.distance_factor = 500.0
        self.target_distance = 1.0
        self.sensitivity = 20

        # Constantly track blue object
        self.blue_detected = False
        self.blue_area = 0
        self.blue_center = (0, 0)
        self.orienting_start_time = None

        # Immediately send a single navigation goal that is a strategic position to visualise all colours. 
        self.send_nav_goal(1.23, -10.2, 0.0)

    def send_nav_goal(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        # Convert from yaw (which is used in the nav2goal) to quaternion
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Sending Nav2 goal: (x={x}, y={y}, yaw={yaw})")
        self.action_client.wait_for_server()
        self.currently_navigating = True
        send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    # This is called after the Nav2Goal action is sent, which accepts or rejects the goasl based on whether it is a valid input or not. 
    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        # If NOT accepted
        if not self.goal_handle.accepted:
            self.get_logger().info("Nav2 goal rejected.")
            self.currently_navigating = False
            return
        # If accepted
        self.get_logger().info("Nav2 goal accepted.")
        self.get_result_future = self.goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    # This function is called after the robot navigates to the required position.
    def get_result_callback(self, _):
        self.currently_navigating = False
        self.get_logger().info("Nav2 goal finished.")

    def feedback_callback(self, _):
        pass

    # Check for green, red, and blue in the camera.
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Determine the masks of the colours.
        blue_mask = self.make_mask(hsv, 120)
        red_mask = self.make_red_mask(hsv)
        green_mask = self.make_mask(hsv, 60)

        # Detect the largest contours for any coloured object.
        found_red, _, _ = self.find_largest_contour(red_mask)
        found_green, _, _ = self.find_largest_contour(green_mask)
        found_blue, area_blue, center_blue = self.find_largest_contour(blue_mask)

        # If the colour detected is red or green, reject the goal and keep the current function running.
        if found_red:
            self.get_logger().info("RED detected, goal rejected.")
        if found_green:
            self.get_logger().info("GREEN detected, goal rejected.")

        self.blue_detected = found_blue
        self.blue_area = area_blue
        self.blue_center = center_blue

        # If in NAVIGATING mode and blue is found, switch from NAVIGATING (cancel current navigation) to ORIENTING mode (run until blue is found or time limit is elapsed).
        if self.mode == "NAVIGATING" and self.blue_detected:
            self.get_logger().info("Blue detected! Cancelling nav and orienting.")
            if self.currently_navigating and self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            self.currently_navigating = False
            self.mode = "ORIENTING"
            self.orienting_start_time = time.time()

        # When in ORIENTING mode, keep orienting until blue is centered within the tolerance specified in the parameters.
        if self.mode == "ORIENTING":
            self.handle_orienting(frame)
        # After blue is centred, the mode will become APPROACHING, which will approach the robot towards the blue box and stop when 1 meter away.
        elif self.mode == "APPROACHING":
            self.handle_approaching()

        cv2.imshow("Camera", frame)
        cv2.waitKey(3)

    # Run until blue is centered within the tolerance range and switch to APPROACHING mode.
    def handle_orienting(self, frame):
        h, w, _ = frame.shape
        midpoint = w // 2
        bx, _ = self.blue_center
        offset = bx - midpoint

        if not self.blue_detected:
            # Keep rotating in place if blue is lost
            self.send_cmd_vel(0.0, self.rotate_speed)
            return

        if abs(offset) < self.center_tolerance:
            # Blue is centered
            self.send_cmd_vel(0.0, 0.0)
            self.mode = "APPROACHING"
            self.get_logger().info("Blue centered. Switching to APPROACHING.")
            return

        elapsed = time.time() - self.orienting_start_time
        if elapsed > self.orientation_timeout:
            # Time limit reached, approach anyway
            self.send_cmd_vel(0.0, 0.0)
            self.mode = "APPROACHING"
            self.get_logger().info("Orientation timeout. Approaching anyway.")
        else:
            # Turn in place
            self.send_cmd_vel(0.0, self.rotate_speed)

    def handle_approaching(self):
        # Keep moving forward until the blue box is 1 meter away.
        if not self.blue_detected or self.blue_area < 1:
            # If we lose it, stop
            self.send_cmd_vel(0.0, 0.0)
            self.get_logger().info("Lost blue object. Stopping.")
            return

        distance_est = self.distance_factor / math.sqrt(self.blue_area)
        if distance_est <= self.target_distance:
            # Reached the blue object (1m away).
            self.send_cmd_vel(0.0, 0.0)
            self.mode = "GOAL_REACHED"
            self.get_logger().info("Close to blue. Goal reached.")
        else:
            self.send_cmd_vel(self.forward_speed, 0.0)

    def send_cmd_vel(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)

    # Create the mask for green and blue
    def make_mask(self, hsv, center_hue):
        lower = np.array([center_hue - self.sensitivity, 100, 100])
        upper = np.array([center_hue + self.sensitivity, 255, 255])
        return cv2.inRange(hsv, lower, upper)

    # Create a split mask for red, as shown in the lab.
    def make_red_mask(self, hsv):
        """Split red hue range into two intervals near 0 and 180."""
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([self.sensitivity, 255, 255])
        lower2 = np.array([180 - self.sensitivity, 100, 100])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask1, mask2)

    # Find the largest contour for the objects. This will be used to calculate the distance from the object.
    def find_largest_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, 0, (0, 0)
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 300:
            return False, 0, (0, 0)
        x, y, w, h = cv2.boundingRect(c)
        center = (x + w // 2, y + h // 2)
        return True, area, center


def main(args=None):
    rclpy.init(args=args)
    node = BlueApproachNode()

    def signal_handler(sig, frame):
        node.get_logger().info("Shutting down.")
        node.send_cmd_vel(0.0, 0.0)
        rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except ROSInterruptException:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
