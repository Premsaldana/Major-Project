import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import os
from flask import Flask, request, jsonify, render_template
import io
# Removed numpy import as it's no longer needed for color check

# --- Configuration ---
MODEL_PATH = 'my_spine_classifier_improved.pth'
CLASS_LABELS = ['normal', 'misalignment', 'fracture', 'degeneration']
NUM_CLASSES = 4
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Detect hardware
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# --- 1. Define Image Transformations ---
data_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    # transforms.Normalize(...) # Add if you used specific normalization in training
])

# --- 2. Function to Load the Model (UPDATED FOR RESNET50) ---
def load_trained_model(model_path, num_classes):
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print("CRITICAL ERROR: Model file not found.")
        return None

    try:
        # Based on your error logs, the weights belong to a ResNet50
        model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle case where checkpoint is a dictionary containing state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("SUCCESS: ResNet50 model loaded successfully.")
        return model
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

# --- 3. Function to Make a Prediction ---
def predict_xray(image_bytes, model, class_labels):
    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes))

        # --- REVERTED PREPROCESSING ---
        # Robustly convert ANY loaded image to RGB. 
        # For X-rays (grayscale), this converts L->RGB (stacking L three times).
        # For photos (color), this converts RGB->RGB.
        img = img.convert('RGB')
        # --- END REVERTED PREPROCESSING ---

        # Preprocess
        image_tensor = data_transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(image_tensor)

        # Calculate probabilities
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_index = torch.max(probabilities, 0)
        
        predicted_class_name = class_labels[predicted_index.item()]
        confidence_score = confidence.item()

        # Format all probabilities
        all_probs = {class_labels[i]: float(probabilities[i].item()) for i in range(len(class_labels))}

        return {
            'predicted_class': predicted_class_name,
            'confidence': confidence_score,
            'all_probabilities': all_probs
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        # Use the generic but necessary error message for file corruption/format issues
        return {'error': 'Error processing file. Please ensure you upload a valid JPEG/PNG image.'}

# --- 4. Load Model ONCE on Server Start ---
loaded_model = load_trained_model(MODEL_PATH, NUM_CLASSES)

# --- 5. Set up Flask Web Server ---
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if loaded_model is None:
        return jsonify({'error': 'Model failed to load on server start'}), 500
    
    if file:
        try:
            img_bytes = file.read()
            results = predict_xray(img_bytes, loaded_model, CLASS_LABELS)
            # Check if predict_xray returned an error dictionary
            if 'error' in results:
                # Return the error message and a 400 status code
                return jsonify(results), 400
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)