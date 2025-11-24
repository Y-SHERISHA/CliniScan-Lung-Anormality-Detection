import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from ultralytics import YOLO
import timm
from torchvision.models.feature_extraction import create_feature_extractor
import os
import gdown

# Page configuration
st.set_page_config(
    page_title="🩻 CliniScan - Lung Abnormality Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------------------------------------------------------

DETECTION_MODEL_ID = "1IzK4Y-wKDSLjLNUGv2tNDiaRyv0W2Iy6"
CLASSIFICATION_MODEL_ID = "1Ao7o8DekO26M9DcsbwVVwTF3Fwkm7o6S"

@st.cache_resource
def download_models():
    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)
    
    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"
    
    if not os.path.exists(det_path):
        with st.spinner("Downloading detection model..."):
            try:
                url = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
                gdown.download(url, det_path, quiet=False)
                st.success("Detection model ready.")
            except Exception as e:
                st.error(f"Error: {e}")
                return False, False

    if not os.path.exists(clf_path):
        with st.spinner("Downloading classification model..."):
            try:
                url = f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}"
                gdown.download(url, clf_path, quiet=False)
                st.success("Classification model ready.")
            except Exception as e:
                st.error(f"Error: {e}")
                return True, False

    return True, True

# Download models
det_ready, clf_ready = download_models()

# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes, drop_rate=dropout)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_classification_model():
    if not clf_ready:
        return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EfficientNetClassifier(num_classes=2, dropout=0.3).to(device)
        model_path = "models/classification/best_clf_model.pth"
        if not os.path.exists(model_path):
            st.error("Classification model not found.")
            return None
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            st.info(f"ModelLoaded | Accuracy: {checkpoint['acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error: {e}")
        return None

@st.cache_resource
def load_detection_model():
    if not det_ready:
        return None
    try:
        model_path = "models/detection/best.pt"
        if not os.path.exists(model_path):
            st.error("Detection model not found.")
            return None
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error: {e}")
        return None

clf_model = load_classification_model()
det_model = load_detection_model()

# -----------------------------------------------------------------------------
# GRAD-CAM FOR EFFICIENTNET-B3
# -----------------------------------------------------------------------------

def generate_gradcam(model, img_tensor):
    if model is None:
        return None, None
    try:
        device = next(model.parameters()).device
        model.eval()
        feature_extractor = create_feature_extractor(
            model.model,
            {"conv_head": "feat"}
        )
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(device)
            out = feature_extractor(img_tensor)
            preds = model(img_tensor)
            pred_class = preds.argmax(dim=1).item()
        feat_map = out["feat"].squeeze().detach().mean(dim=0).cpu().numpy()
        heatmap = cv2.resize(feat_map, (512, 512))
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return heatmap, pred_class
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None, None

# -----------------------------------------------------------------------------
# INTERFACE & UI
# -----------------------------------------------------------------------------

st.title("🩻 CliniScan: AI-powered Detection")
st.markdown("""
Effortlessly analyze chest X-rays for lung abnormalities.<br>
**Upload an image**, review AI findings, and explore visual explanations.<br>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About CliniScan")
    st.markdown(
        """
        Detectable conditions (14 types):  
        • Aortic enlargement  
        • Atelectasis  
        • Calcification  
        • Cardiomegaly  
        • Consolidation  
        • ILD  
        • Infiltration  
        • Lung Opacity  
        • Nodule/Mass  
        • Other lesion  
        • Pleural effusion  
        • Pleural thickening  
        • Pneumothorax  
        • Pulmonary fibrosis  
        
        **Classes:**  
        • Abnormal  
        • Normal  
        --
        ⚠ For educational/research use only.
        """
    )

st.markdown("---")
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("📷 Uploaded X-ray")
    st.image(image, use_column_width=True)
    
    if clf_model is None or det_model is None:
        st.error("Model loading issue. Check credentials/model weights.")
        st.stop()
    
    st.markdown("---")
    col1, col2 = st.columns([2,2])
    
    with col1:
        st.subheader("🔍 Overall Classification")
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image)
        device = next(clf_model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            preds = clf_model(img_tensor.unsqueeze(0))
            probs = torch.nn.functional.softmax(preds, dim=1)
            pred_class = torch.argmax(probs).item()
        
        class_names = ["Abnormal", "Normal"]
        
        st.markdown(f"**Prediction:** {class_names[pred_class]}")
        st.markdown(f"**Estimated confidence:** {probs[0][pred_class]:.2%}")
        st.write("**Class distribution:**")
        for i, name in enumerate(class_names):
            st.write(f"{name}: {probs[0][i].item():.2%}")
            st.progress(float(probs[0][i].item()))
        
        st.markdown("---")
        st.subheader("🧠 Grad-CAM Visual Explanation")
        st.markdown("Highlighted regions indicate areas that influenced the prediction (red/yellow is high focus).")
        heatmap, _ = generate_gradcam(clf_model, img_tensor)
        if heatmap is not None:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            original_resized = np.array(image.resize((512, 512)))
            overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
            st.image(overlay, caption="Grad-CAM focus", use_column_width=True)
        else:
            st.warning("Unable to generate Grad-CAM at this time.")
    
    with col2:
        st.subheader("📦 Detected Abnormalities")
        with st.spinner("Scanning for abnormalities..."):
            results = det_model.predict(np.array(image), conf=0.25, verbose=False)
        res_img = results[0].plot()
        st.image(res_img, caption="Detected regions (Bounding Boxes)", use_column_width=True)
        
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.markdown("**Top findings:**")
            for i in range(min(5, len(boxes))):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                st.write(f"{i+1}. {det_model.names[cls_id]} ({conf:.2%} confidence)")
                st.progress(conf)
            st.write(f"Total detections: {len(boxes)}")
            st.write(f"Average confidence: {float(boxes.conf.mean()):.2%}")
        else:
            st.success("No significant abnormalities detected. This X-ray appears within normal limits per AI analysis.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 16px;'>
<strong>⚠ DISCLAIMER:</strong>  
This web application is intended for educational and research purposes only.<br>
Do NOT use these results for clinical diagnosis or medical decision-making.<br>
Consult a licensed healthcare professional for all medical interpretations.<br>
<hr>
Y Sherisha  
<a href='https://github.com/Y-SHERISHA/CliniScan-Lung-Anormality-Detection'>GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
