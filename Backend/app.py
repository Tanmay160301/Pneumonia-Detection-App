from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS extension

import cv2
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# def process_image(image_path, model_path, threshold=0.5):
#     # Load the trained model
#     model = load_model(model_path)

#     # Resize the image to the required dimensions
#     img_size = 150
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     resized_img = cv2.resize(img, (img_size, img_size))
    
#     # Normalize and reshape the image for the model
#     processed_img = resized_img / 255.0
#     processed_img = processed_img.reshape(-1, img_size, img_size, 1)

#     # Make a prediction using the loaded model
#     prediction = model.predict(processed_img)

#     # Classify as pneumonia (1) or not pneumonia (0) based on the model's output
#     result = 1 if prediction < threshold else 0

#     return result

# def process_image(image_path, model_path, threshold=0.5):
#     # Load the trained model
#     model = load_model(model_path)

#     # Resize the image to the required dimensions
#     img_size = 150
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     resized_img = cv2.resize(img, (img_size, img_size))
    
#     # Normalize and reshape the image for the model
#     processed_img = resized_img / 255.0
#     processed_img = processed_img.reshape(-1, img_size, img_size, 1)

#     # Make a prediction using the loaded model
#     prediction = model.predict(processed_img)

#     # Classify as pneumonia (1) or not pneumonia (0) based on the model's output
#     result = 1 if prediction < threshold else 0

#     return result

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

    # Log the prediction for debugging
    print('Prediction:', prediction)

    # Classify as pneumonia (1) or not pneumonia (0) based on the model's output
    result = 1 if prediction[0][0] < threshold else 0

    return result

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']

    # Save the image to a temporary file
    temp_path = 'temp_image.jpg'
    image_file.save(temp_path)

    # Process the image using the provided function
    model_path = 'C:\\Users\\Admin\\Desktop\\PD\\pneumonia_detection_model.h5'
    binary_result = process_image(temp_path, model_path)

    # Log the binary result for debugging
    print('Binary Result:', binary_result)

    # Return the result as JSON
    return jsonify({'result': 'Pneumonia Detected' if binary_result else 'No Pneumonia'}), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
