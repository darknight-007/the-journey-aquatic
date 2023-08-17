import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import torch
import torchvision.transforms as T
from PIL import Image as PILImage
import numpy as np
import matplotlib.pyplot as plt
import requests
import time


class LEDScreenCalibrator(Node):
    def __init__(self):
        super().__init__('depth_estimator')

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Adjust the topic name as needed
            self.image_callback,
            10
        )
        self.timer_ = self.create_timer(0.1, self.calibrate_leds)

        self.subscription  # prevent unused variable warning
        self.depth_publisher = self.create_publisher(Image, '/spots', 10)
        self.mapped_image_publisher = self.create_publisher(Image, '/mapped_image', 10)
        self.mapped_pixel_publisher = self.create_publisher(Image, '/mapped_pixels', 10)
        self.bridge = CvBridge()
        self.url2 = "http://192.168.0.122/"
        self.params = {
            "strip": "0",
            "pixel": "0",
            "red": "0",
            "blue": "255",
            "green": "0",
            "brightness": "255"
        }
        self.pixel_count = [600, 600, 300]
        self.hop_lengths = [2, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        self.pixel_idx = 0
        self.strip_idx = 0
        self.hop_length = self.hop_lengths[0]
        self.light_map = {}
        self.mapped_bright_spots = []
        self.mapped_pixels = []
        self.do_tracking = True
        self.X = np.ndarray(shape=(800,2), dtype=int)
        self.Y = np.ndarray(shape=(800,2), dtype=int)

    def calibrate_leds(self):
        print(self.strip_idx, self.pixel_idx)
        if self.do_tracking:
            self.params["red"] = "0"  # Send the request to the LED strip
            self.params["green"] = "255"
            self.params["blue"] = "0"
            if self.strip_idx > 1:
                self.strip_idx = 0
            if (self.pixel_idx > self.pixel_count[self.strip_idx]):
                self.strip_idx = self.strip_idx + 1
                self.pixel_idx = 0
            # print(self.strip_idx, self.pixel_idx)
            self.params["pixel"] = str(
                self.pixel_idx % self.pixel_count[self.strip_idx])  # Send the request to the LED strip
            self.params["strip"] = str(self.strip_idx % 2)  # Send the request to the LED strip
            self.pixel_idx = self.pixel_idx + self.hop_length

            response = requests.get(self.url2, params=self.params)
            time.sleep(0.1)

            # time.sleep(0.1)
            self.params["red"] = "0"  # Send the request to the LED strip
            self.params["green"] = "0"
            self.params["blue"] = "0"
            response = requests.get(self.url2, params=self.params)
            time.sleep(0.01)


        elif self.pixel_idx < len(self.light_map.keys()):
            self.params["red"] = "0"  # Send the request to the LED strip
            self.params["green"] = "20"
            self.params["blue"] = "10"
            self.params["brightness"] = "50"

            keys = list(self.light_map.keys())
            print(keys)
            key = keys[self.pixel_idx]
            tokens = key.split(",")
            x_val = int(tokens[0])
            y_val = int(tokens[1])

            value = self.light_map[key]
            print(str(value[0]) + " " + str(value[1]))
            self.X[self.pixel_idx,:]=[x_val, y_val]
            self.Y[self.pixel_idx,:]=value
            self.params["pixel"] = str(value[1])
            self.params["strip"] = str(value[0])
            response = requests.get(self.url2, params=self.params)
            self.pixel_idx = self.pixel_idx + 1
            # print(self.X, self.Y)
            # time.sleep(0.1)
        else:
            np.savez('array_data122_aug16_2.npz', name1=self.X, name2=self.Y)
            exit(0)





    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)
        threshold_value = 160  # You can adjust this threshold value based on your image
        _, thresholded_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_with_bright_spots = cv_image.copy()

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(image_with_bright_spots, center, radius, (0, 255, 0), 2)  # Green circle

        if self.do_tracking:
            if len(contours) > 0:
                (x, y), radius = cv2.minEnclosingCircle(contours[0])
                center = (int(x), int(y))
                radius = int(radius)
                self.light_map[str(int(x)) + "," + str(int(y))] = [self.strip_idx, self.pixel_idx]
                print(len(self.light_map.keys()))

                # print(len(self.light_map), str(self.strip_idx) + ',' + str(self.pixel_idx), center)
                if len(self.mapped_bright_spots) == 0:
                    self.mapped_bright_spots = cv_image.copy()
                if len(self.mapped_pixels) == 0:
                    self.mapped_pixels = cv_image.copy()
                cv2.circle(self.mapped_bright_spots, center, radius, (0, 255, 0), 2)  # Green circle
                cv2.circle(self.mapped_pixels, (int(self.pixel_idx + 3) * 1, int(self.strip_idx + 2) * 20), radius,
                           (200, 200, 0), 1)  # Green circle
                val = self.light_map[str(int(x)) + "," + str(int(y))]
                cv2.circle(self.mapped_bright_spots, (val[0], val[1]), 10, (255, 0, 255), 2)  # Green circle
                modified_mapped_image_msg = self.bridge.cv2_to_imgmsg(self.mapped_bright_spots, encoding='bgr8')
                self.mapped_image_publisher.publish(modified_mapped_image_msg)
                self.mapped_pixel_publisher.publish(self.bridge.cv2_to_imgmsg(self.mapped_pixels, encoding='bgr8'))

        if self.strip_idx == 1 and self.pixel_idx > 300:
            self.do_tracking = False
            self.strip_idx =0
            self.pixel_idx=0

        modified_image_msg = self.bridge.cv2_to_imgmsg(image_with_bright_spots, encoding='bgr8')
        self.depth_publisher.publish(modified_image_msg)


def find_nearest_indices(target_vector, vectors_set, k=5):
    # Calculate Euclidean distances between target_vector and all vectors in vectors_set
    distances = np.linalg.norm(vectors_set - target_vector, axis=1)

    # Sort indices of vectors based on distances
    nearest_indices = np.argsort(distances)

    # Get the indices of the k nearest vectors
    k_nearest_indices = nearest_indices[:k]


    return k_nearest_indices

def main(args=None):
    rclpy.init()
    node = LEDScreenCalibrator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
