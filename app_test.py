import torch
import torch.nn as nn
import timm
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import io

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # This allows your HTML file to talk to this Python script

# --- 1. Define The Model Structure (Must match training exactly) ---
class ECBHybrid(nn.Module):
    def __init__(self, num_classes):
        super(ECBHybrid, self).__init__()
        # Initialize pretrained models
        self.vit_encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        self.cnn_encoder = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)

    def forward(self, x):
        vit_out = self.vit_encoder(x)
        cnn_out = self.cnn_encoder(x)
        return vit_out, cnn_out

# --- 2. Configuration & Load Model ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Normal', 'Misalignment', 'Fracture', 'Degeneration'] # Class order matches training folders
MODEL_PATH = 'final_ecb_hybrid_model_state1.pth'

# Prepare model
try:
    model = ECBHybrid(num_classes=4)
    # Load weights onto the correct device
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval() # Set to evaluation mode (important!)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- 3. Image Transformation Pipeline ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # 1. Read and Transform Image
        file = request.files['image']
        img = Image.open(file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # 2. Get Prediction
        with torch.no_grad():
            # Based on your notebook, we use the CNN head for the final decision
            _, cnn_out = model(img_tensor)
            
            # Apply Softmax to get percentages
            probabilities = torch.nn.functional.softmax(cnn_out[0], dim=0)
            confidence, predicted_index = torch.max(probabilities, 0)
            
            # 3. Format Response
            diagnosis = CLASSES[predicted_index.item()]
            
            response = {
                'diagnosis': diagnosis,
                'confidence': f"{confidence.item() * 100:.2f}%",
                'breakdown': {
                    'Normal': f"{probabilities[0].item() * 100:.2f}%",
                    'Misalignment': f"{probabilities[1].item() * 100:.2f}%",
                    'Fracture': f"{probabilities[2].item() * 100:.2f}%",
                    'Degeneration': f"{probabilities[3].item() * 100:.2f}%",
                }
            }
            
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    # Add this to app.py so you don't get a 404 error if you visit the link
@app.route('/', methods=['GET'])
def home():
    return "✅ Spine Analysis Server is RUNNING. Please open index.html to use the app."

if __name__ == '__main__':
    print("Starting Spine Analysis Server...")
    app.run(debug=True, port=5000)