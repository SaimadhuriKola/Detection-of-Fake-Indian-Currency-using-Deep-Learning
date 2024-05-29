import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

# Load your DL model
model = tf.keras.models.load_model('C:/Users/bobba/Downloads/indian-currency-classification-master/indian-currency-classification-master/rfclassifier_600.sav')

def preprocess_image(image):
    # Resize image to match input size expected by the model
    image = image.resize((224, 224))
    # Convert image to numpy array
    img_array = np.array(image)
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    # Add batch dimension and return
    return np.expand_dims(img_array, axis=0)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Print the received image file
    print(request.files['image'])
    
    image = request.files['image']
    
    try:
        # Open and preprocess the image
        img = Image.open(image)
        img_array = preprocess_image(img)
        
        # Perform inference with the model
        predictions = model.predict(img_array)
        # Assuming binary classification (fake vs real)
        # Replace this with your actual prediction logic
        class_label = "Fake" if predictions[0][0] > 0.5 else "Real"

        # Log the predicted class label
        print("Predicted class:", class_label)
        
        # Return the classification result
        return jsonify({'result': class_label}), 200
    
    except Exception as e:
        # Log any errors
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
