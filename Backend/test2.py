import subprocess
import sys

# Name of your Conda environment
conda_environment = 'tf'

# Activate Conda environment (change the command based on your OS)
activate_command = f'conda activate {conda_environment}'
subprocess.run(activate_command, shell=True, executable='/bin/bash' if 'linux' in sys.platform else None)

# Now you can run your code
import cv2
import tensorflow as tf
from keras.models import load_model

def process_image(image_path, model_path, threshold=0.5):
    # Load the trained model
    model = load_model(model_path)

    # Resize the image to the required dimensions
    img_size = 150
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (img_size, img_size))
    
    # Normalize and reshape the image for the model
    processed_img = resized_img / 255.0
    processed_img = processed_img.reshape(-1, img_size, img_size, 1)

    # Make a prediction using the loaded model
    prediction = model.predict(processed_img)

    # Classify as pneumonia (1) or not pneumonia (0) based on the model's output
    result = 1 if prediction < threshold else 0

    return result

# Example usage
image_path = 'C:\\Users\\Admin\\Desktop\\PD\\Internet3.jpg'  # Replace with the actual path to your image
model_path = 'C:\\Users\\Admin\\Desktop\\PD\\pneumonia_detection_model.h5'

binary_result = process_image(image_path, model_path)
print(f"Binary Result: {'Pneumonia Detected' if binary_result else 'No Pneumonia'}")
