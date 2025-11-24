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

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="🫁 CliniScan - Chest X-ray Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL DOWNLOAD ---
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

# --- MODEL LOADING ---
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

# --- GRAD-CAM ---
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

# --- SIDEBAR (Detectable Conditions section removed) ---
with st.sidebar:
    st.header("🫁 About CliniScan")
    st.markdown("""
    <div style="font-size:18px;">
    <strong>Features:</strong><br>
    🔎 <b>Fast Chest X-ray Analysis</b><br>
    ⚡ Detects many abnormalities<br>
    🎯 Visual explanations (Grad-CAM)<br>
    <hr>
    <strong>Classes:</strong> <span style="color:green;">Normal</span>, <span style="color:red;">Abnormal</span>
    <hr>
    <span style="color:orange;">⚠ For educational, research, and non-clinical use only!</span>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN HEADER ---
st.markdown("<h1 style='text-align:center;'>🫁 CliniScan: Chest X-ray Analyzer</h1>", unsafe_allow_html=True)

# --- Detectable Conditions (NEW BLOCK) ---
st.markdown("""
<div style='border: 1px solid #ddd; border-radius: 8px; background: #fafafa; padding: 16px; margin-bottom: 20px;'>
<h3 style='text-align:center;'>✨ Detectable Conditions</h3>
<ul style="font-size:17px; margin-left: 30px;">
  <li>Aortic enlargement</li>
  <li>Atelectasis</li>
  <li>Calcification</li>
  <li>Cardiomegaly</li>
  <li>Consolidation</li>
  <li>Interstitial Lung Disease (ILD)</li>
  <li>Infiltration</li>
  <li>Lung Opacity</li>
  <li>Nodule / Mass</li>
  <li>Other lesion</li>
  <li>Pleural effusion</li>
  <li>Pleural thickening</li>
  <li>Pneumothorax</li>
  <li>Pulmonary fibrosis</li>
</ul>
</div>
""", unsafe_allow_html=True)

# --- Upload Message and rest of your app follows ---
st.markdown("""
<div style='text-align:center; font-size:18px;'>
🔄 <b>Upload a chest X-ray image below to begin detection!</b><br>
Supported formats: JPG, JPEG, PNG
</div>
""", unsafe_allow_html=True)

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

    # --- MAIN LAYOUT ---
    st.markdown("<style>div[data-testid=\"stHorizontalBlock\"] div:first-child {text-align: center;}</style>", unsafe_allow_html=True)
    main_col1, main_col2 = st.columns([2,3])

    with main_col1:
        st.markdown("<h2 style='text-align:center;'>🧑‍⚕️ Model Confidence & Prediction</h2>", unsafe_allow_html=True)
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

        st.write(f"**Prediction:** {class_names[pred_class]}")
        st.write(f"**Estimated confidence:** {probs[0][pred_class]:.2%}")
        st.write("**Class distribution:**")
        for i, name in enumerate(class_names):
            st.write(f"{name}: {probs[0][i].item():.2%}")
            st.progress(float(probs[0][i].item()))

        # Grad-CAM
        st.markdown("---")
        st.markdown("<h3 style='text-align:center;'>🧠 Grad-CAM Insights</h3>", unsafe_allow_html=True)
        st.markdown("Highlighted regions indicate areas that influenced the prediction (red/yellow is high focus).")
        heatmap, _ = generate_gradcam(clf_model, img_tensor)
        if heatmap is not None:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            original_resized = np.array(image.resize((512, 512)))
            overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
            st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)
        else:
            st.warning("Unable to generate Grad-CAM at this time.")

    with main_col2:
        st.markdown("<h2 style='text-align:center;'>🩺 Detected Problems</h2>", unsafe_allow_html=True)
        with st.spinner("Scanning for abnormalities..."):
            results = det_model.predict(np.array(image), conf=0.25, verbose=False)
        res_img = results[0].plot()
        st.image(res_img, caption="Detected regions (Bounding Boxes)", use_column_width=True)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.markdown("#### Top findings:")
            findings_md = "| # | Condition | Confidence | Status |\n|---|---|---|---|\n"
            def color_icon(conf):
                return "🟢" if conf > 0.7 else ("🟡" if conf > 0.4 else "🔴")
            for i in range(min(5, len(boxes))):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                findings_md += f"| {i+1} | {det_model.names[cls_id]} | {conf:.2%} | {color_icon(conf)} |\n"
            st.markdown(findings_md)
            st.write(f"Total detections: {len(boxes)}")
            st.write(f"Average confidence: {float(boxes.conf.mean()):.2%}")
        else:
            st.success("No significant abnormalities detected. This X-ray appears within normal limits per AI analysis.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 16px;'>
<strong>🛑 DISCLAIMER:</strong><br>
This app is for research and educational use only.<br>
Medical decisions should always be made by qualified professionals.<br>
<hr>
Developed by Y Sherisha <br>
<a href='https://github.com/Y-SHERISHA/CliniScan-Lung-Anormality-Detection'>GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
