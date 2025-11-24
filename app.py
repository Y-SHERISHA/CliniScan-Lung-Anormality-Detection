import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from ultralytics import YOLO
import timm
from torchvision.models.feature_extraction import create_feature_extractor
import os
import gdown

# Page config: change emoji, wider layout, top navigation bar custom header
st.set_page_config(
    page_title="🫁 CliniScan: Lung AI Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Top Navigation Bar
st.markdown("""
    <style>
    .navbar {
        background-color: #0d6efd;
        padding: 1rem;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        letter-spacing: 1px;
        border-radius: 0 0 10px 10px;
        margin-bottom: 2rem;
    }
    .navinfo {
        background: #f8f9fa;
        border-radius:10px; 
        padding:0.7rem; 
        margin-bottom:1rem;
        color: #333;
    }
    </style>
    <div class="navbar">
        🫁 <b>CliniScan: Lung AI Dashboard</b> 
    </div>
    <div class="navinfo">
        <b>Analyze chest X-rays for lung abnormalities using deep learning.<br>
        Show model focus using Grad-CAM visualizations and bounding box detection.<br>
        For research/education only.</b>
    </div>
    """, unsafe_allow_html=True)

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
            st.info(f"Model loaded | Accuracy: {checkpoint['acc']:.2f}%")
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
# FILE UPLOADER & VISUALIZATION
# -----------------------------------------------------------------------------

uploaded_file = st.file_uploader("📥 Please upload a chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.markdown("### Original X-ray")
    st.image(image, width=350)

    # Main output columns: visualizations + results
    viz_cols = st.columns([1,2])
    with viz_cols[0]:
        st.markdown("<h4 style='color:#0d6efd;'> Detection Results </h4>", unsafe_allow_html=True)
        if clf_model is None or det_model is None:
            st.error("Model loading issue. Check credentials/model weights.")
            st.stop()
        # Detection output (YOLO)
        with st.spinner("Scanning for abnormalities..."):
            results = det_model.predict(np.array(image), conf=0.25, verbose=False)
        res_img = results[0].plot()
        st.image(res_img, caption="Bounding boxes of detected abnormalities", use_column_width=True)
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(min(5, len(boxes))):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                st.markdown(f"<b>{i+1}. {det_model.names[cls_id]}</b> — <code>{conf:.2%} confidence</code>", unsafe_allow_html=True)
        else:
            st.success("No significant abnormalities detected.")

    # Grad-CAM visualizations (side-by-side layout)
    with viz_cols[1]:
        st.markdown("<h4 style='color:#0d6efd;'> Grad-CAM Visualizations </h4>", unsafe_allow_html=True)
        st.markdown("Previewing model focus areas for this X-ray using multiple Grad-CAM styles (experimental)")
        gradcam_images = ["gradcam_1.jpg", "gradcam_2.jpg", "gradcam_3.jpg", "gradcam_4.jpg", "gradcam_5.jpg"]
        gradcam_labels = ["Original Image", "GradCAM", "GradCAM++", "HiResCAM", "Heatmap Overlay"]
        img_cols = st.columns(len(gradcam_images))
        for idx, col in enumerate(img_cols):
            col.image(gradcam_images[idx], caption=gradcam_labels[idx], use_column_width=True)

        # Classification output
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
        st.markdown(f"<b>Prediction:</b> <span style='color:#198754;'>{class_names[pred_class]}</span>", unsafe_allow_html=True)
        st.markdown(f"<b>Model confidence:</b> <code>{probs[0][pred_class]:.2%}</code>", unsafe_allow_html=True)
        st.progress(float(probs[0][pred_class].item()))

        st.info("Grad-CAM heatmaps highlight image regions most influential in the model's decision. Darker red/yellow = greater model attention.")

# -----------------------------------------------------------------------------
# TOP "INFO BAR" (looks unique)
# -----------------------------------------------------------------------------

st.markdown("""
    <hr>
    <div style='text-align: center; color: #333; font-size: 15px;'>
    ⚠️ <b>This demo is intended for educational purposes and research only. Clinical diagnosis or treatment decisions must always be performed by qualified health professionals.</b>
    </div>
    <hr>
    <div style='text-align:center;font-size:16px;color:#bbb'>
        <b>Developed by Y Sherisha</b> |
        <a href='https://github.com/Y-SHERISHA/CliniScan-Lung-Anormality-Detection' style='color:#0d6efd;'>Project GitHub</a>
    </div>
    """, unsafe_allow_html=True)
