import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# Load an image and convert it to a NumPy array
def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)


# Save NumPy array as image
def save_image(image_array, output_path):
    image = Image.fromarray(image_array)
    image.save(output_path)


# Convert image to grayscale
def convert_to_grayscale(image_array):
    return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.unit8)
