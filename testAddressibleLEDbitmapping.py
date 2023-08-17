import numpy as np
import random
import math
from rclpy.node import Node as rcl_node
from PIL import Image as PILImage
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
from scipy.ndimage import distance_transform_edt

class LEDScreenCalibrator():
    def __init__(self):
        # data = np.load('/home/dreamslab/PycharmProjects/testCopiloting/array_data187.npz')
        data = np.load('/home/dreamslab/PycharmProjects/testCopiloting/array_data122_aug16_2.npz')

        self.UV = data['name1']
        self.SP = data['name2']
        self.params = {
            "strip": "0",
            "pixel": "0",
            "red": "0",
            "blue": "0",
            "green": "0",
            "brightness": "2"
        }
        self.url = "http://192.168.0.122/"
        # sorted_indices = np.argsort(self.UV[:,1])[::-1]
        self.UV = self.UV[np.all(self.UV != 0, axis=1)]
        self.A, self.B, self.C = 1, 0, -640  # Example line equation: x - 320 = 0

        # Sample usage
        # image_path = "/home/dreamslab/Downloads/box-pattern.png"
        image_path = "/home/dreamslab/corals-pattern.jpg"

        self.binary_mask, self.image_resized = self.image_to_binary_mask(image_path)

        # If you want to visualize the mask using PIL:
        PILImage.fromarray((self.binary_mask * 255).astype(np.uint8)).show()

        for i in range(500):
            desired_coord = [320, 240]
            distances_sq = []

            # if i % 13 == 0:
            #     distances_sq = np.sum((self.UV - desired_coord) ** 2, axis=1)
            # elif i % 3 == 0:
            for uv in self.UV:
                distances_sq.append(self.distance_to_mask(uv, self.binary_mask))
                print("bitmasking", uv)
            # else:
            #     for uv in self.UV:
            #         distances_sq.append(self.distance_to_line(self.A, self.B, self.C, uv))
            #         distances_sq = np.array(distances_sq)


            sorted_indices = np.argsort(distances_sq)
            self.params["red"] = str(random.randint(0, 255))  # Send the request to the LED strip
            self.params["green"] = str(random.randint(0, 255))
            self.params["blue"] = str(random.randint(0, 255))

            for sorted_index in sorted_indices:
                sp = self.SP[sorted_index]
                uv = self.UV[sorted_index]
                print(self.UV[sorted_index],sp)
                self.params["pixel"] = str(sp[1])
                self.params["strip"] = str(sp[0])
                r, g, b = self.image_resized.getpixel((uv[0], uv[1]))
                self.params["red"] = str(r)
                self.params["green"] = str(g)
                self.params["blue"] = str(b)
                print(self.params)
                print(self.A, self.B, self.C)
                response = requests.get(self.url, params=self.params)
                time.sleep(0.1)
            self.A, self.B, self.C = random.uniform(-1, 1), random.uniform(-1, 1), -random.randint(0, 640)  # Example line equation: x - 320 = 0

    def distance_to_line(self, A, B, C, point):
        x1, y1 = point
        return abs(A * x1 + B * y1 + C) / np.sqrt(A ** 2 + B ** 2)

    def distance_to_circle(self, center_x, center_y, radius, point_x, point_y):
        dx = point_x - center_x
        dy = point_y - center_y
        distance = math.sqrt(dx ** 2 + dy ** 2) - radius
        return distance

    def image_to_binary_mask(self, image_path, threshold=128):
        with PILImage.open(image_path) as image:
            image_resized = image.resize((640, 480))
            grayscale = image_resized.convert('L')
            numpy_array = np.array(grayscale)
            binary_mask = numpy_array > threshold
        return binary_mask, image_resized

    def distance_to_mask(self, point, binary_mask):
        # Compute the Euclidean distance transform of the binary mask
        edt_image = distance_transform_edt(binary_mask)

        # Extract the distance value at the specified point
        distance = edt_image[point[1], point[0]]

        return distance


def main(args=None):
    LEDScreenCalibrator()

if __name__ == '__main__':
    main()
