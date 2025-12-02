import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from flask import Flask, request, jsonify, render_template
import io
import cv2
import numpy as np
import base64
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

# Agg backend for headless server stability
matplotlib.use('Agg') 

# --- Configuration ---
MODEL_PATH_A = 'best_model_convnext.pth' 
MODEL_PATH_B = 'best_model_effnet.pth'

CLASS_LABELS = ['degeneration', 'fracture', 'misalignment', 'normal']
NUM_CLASSES = 4
IMG_HEIGHT = 224
IMG_WIDTH = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Preprocessing (CLAHE) ---
def apply_clahe_cv2(img_np):
    if len(img_np.shape) == 2:
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        img_lab = cv2.cvtColor(img_lab, cv2.COLOR_RGB2LAB)
    else:
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l, a, b = cv2.split(img_lab)
    l_enhanced = clahe.apply(l)
    img_merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(img_merged, cv2.COLOR_LAB2RGB)

data_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. SOTA Grad-CAM++ Implementation ---
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output): self.activations = output
    def save_gradient(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        score_exp = torch.exp(score)
        grads_power_2 = gradients.pow(2)
        grads_power_3 = gradients.pow(3)
        sum_activations = torch.sum(activations, dim=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, :, None, None] * grads_power_3 + eps)
        aij = torch.where(gradients != 0, aij, torch.zeros_like(aij))
        
        weights = torch.maximum(gradients, torch.zeros_like(gradients)) * aij
        weights = torch.sum(weights, dim=(2, 3))

        heatmap = torch.sum(weights[:, :, None, None] * activations, dim=1)
        heatmap = F.relu(heatmap)
        heatmap = heatmap.cpu().detach().numpy()[0]
        
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap

# --- 3. Load Ensemble Models ---
def load_ensemble():
    print("Loading Ensemble Models...")
    try:
        model_a = models.convnext_tiny(weights=None)
        model_a.classifier[2] = nn.Linear(model_a.classifier[2].in_features, NUM_CLASSES)
        model_a.load_state_dict(torch.load(MODEL_PATH_A, map_location=device))
        model_a.to(device).eval()
        target_layer_a = model_a.features[-1] 

        model_b = models.efficientnet_b3(weights=None)
        model_b.classifier[1] = nn.Linear(model_b.classifier[1].in_features, NUM_CLASSES)
        model_b.load_state_dict(torch.load(MODEL_PATH_B, map_location=device))
        model_b.to(device).eval()
        target_layer_b = model_b.features[-1]

        cam_a = GradCAMPlusPlus(model_a, target_layer_a)
        cam_b = GradCAMPlusPlus(model_b, target_layer_b)
        
        print("✅ Ensemble & Grad-CAM++ loaded.")
        return model_a, model_b, cam_a, cam_b
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None, None

model_a, model_b, cam_a, cam_b = load_ensemble()
app = Flask(__name__)

# --- 4. GRAPH GENERATOR ---
def generate_bar_chart(probabilities):
    labels = [l.capitalize() for l in CLASS_LABELS]
    values = [probabilities[l] * 100 for l in CLASS_LABELS]
    
    plt.figure(figsize=(6, 3))
    colors = ['#bdc3c7' if v < max(values) else '#e74c3c' for v in values]
    
    plt.barh(labels, values, color=colors, height=0.6)
    plt.xlabel('Confidence Score (%)')
    plt.title('Differential Diagnosis Probability')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    buff = io.BytesIO()
    plt.savefig(buff, format='png', dpi=100)
    plt.close()
    return base64.b64encode(buff.getvalue()).decode("utf-8")

# --- 5. CLINICAL HEATMAP ---
def create_clinical_heatmap(original_img_np, heatmap):
    h, w = original_img_np.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    
    gray_img = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2GRAY)
    _, bg_mask = cv2.threshold(gray_img, 20, 1, cv2.THRESH_BINARY)
    heatmap = heatmap * bg_mask

    heatmap = np.maximum(heatmap, 0)
    heatmap[heatmap < 0.25] = 0 
    if np.max(heatmap) != 0: heatmap = heatmap / np.max(heatmap)
    
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    
    mask = heatmap_uint8 > 0
    superimposed_img = original_img_np.copy()
    overlay = cv2.addWeighted(heatmap_rgb, 0.5, original_img_np, 0.5, 0)
    superimposed_img[mask] = overlay[mask]
    
    return superimposed_img

@app.route('/')
def home(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_np_original = np.array(pil_img)
        
        img_np_clahe = apply_clahe_cv2(img_np_original)
        pil_img_clahe = Image.fromarray(img_np_clahe)
        img_tensor = data_transform(pil_img_clahe).unsqueeze(0).to(device)

        with torch.no_grad():
            out_a = model_a(img_tensor)
            out_b = model_b(img_tensor)
            probs = (torch.softmax(out_a, 1) + torch.softmax(out_b, 1)) / 2
            prob_vector = probs[0]
            confidence, predicted_index = torch.max(prob_vector, 0)
            predicted_class = CLASS_LABELS[predicted_index.item()]

        heatmap_a = cam_a.generate(img_tensor, predicted_index.item())
        heatmap_b = cam_b.generate(img_tensor, predicted_index.item())
        combined_heatmap = (heatmap_a + heatmap_b) / 2
        
        img_display = cv2.resize(img_np_original, (IMG_WIDTH, IMG_HEIGHT)) 
        overlay = create_clinical_heatmap(img_display, combined_heatmap)
        
        buff = io.BytesIO()
        Image.fromarray(overlay).save(buff, format="PNG")
        heatmap_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

        prob_dict = {CLASS_LABELS[i]: float(prob_vector[i].item()) for i in range(len(CLASS_LABELS))}
        graph_b64 = generate_bar_chart(prob_dict)

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence.item(),
            'all_probabilities': prob_dict,
            'heatmap_image': f"data:image/png;base64,{heatmap_b64}",
            'graph_image': f"data:image/png;base64,{graph_b64}"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)