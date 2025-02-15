import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# Load an image and convert it to a NumPy array
def load_image(image_path):
    image: Image = Image.open(image_path)
    return np.array(image)


# Save NumPy array as image
def save_image(image_array, output_path):
    image = Image.fromarray(image_array)
    image.save(output_path)


# Convert image to grayscale
def convert_to_grayscale(image_array):
    return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


# Apply filter
def apply_filter(image_array):
    kernel = np.array([[-1, -1, -1], [-1, -8, 1], [-1, -1, -1]])
    # Get the dimensions of the image
    height, width = image_array.shape

    # Create an empty output array
    filtered_image = np.zeros_like(image_array, dtype=np.float32)

    # Apply the kernel to the image
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract the 3x3 region around the pixel
            region = image_array[i - 1:i + 2, j - 1:j + 2]
            # Apply the kernel and sum the result
            filtered_image[i, j] = np.sum(region * kernel)

    # Clip the values to the range [0, 255] and convert to uint8
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)


# Log processing results to a csv file
def log_results(input_path, output_path, operation):
    log_data = {
        "Input Image": [input_path],
        "Output Image": [output_path],
        "Operation": [operation]
    }
    df = pd.DataFrame(log_data)
    df.to_csv("processing_log.csv", mode="a", index=False, header=False)


# Main function
def main():
    input_path = "images/input_image.png"
    output_path = "images/processed_image.jpg"

    # Load image
    image_array = load_image(input_path)

    # Display original image
    plt.imshow(image_array)
    plt.title("Original Image")
    plt.show()

    # Convert to grayscale
    grayscale_image = convert_to_grayscale(image_array)
    plt.imshow(grayscale_image, cmap="gray")
    plt.title("Grayscale Image")
    plt.show()

    # Apply filter
    filtered_image = apply_filter(grayscale_image)
    plt.imshow(filtered_image, cmap="gray")
    plt.title("Filtered Image")
    plt.show()

    # Save processed image
    save_image(filtered_image, output_path)

    # Log results
    log_results(input_path, output_path, "Grayscale and Edge Detection")


if __name__ == "__main__":
    main()

