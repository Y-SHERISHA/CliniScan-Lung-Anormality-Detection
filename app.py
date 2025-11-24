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

# ---- Layout and Colors ----
st.set_page_config(page_title="CliniScan | AI Chest X-ray Assistant", layout="wide", initial_sidebar_state="expanded")
MAIN_BG = "#F4F6F7"
ACCENT = "#2B7A78"
NORMAL_C = "#379e61"
ABNORMAL_C = "#d35400"

def colored_text(text, color):
    return f"<span style='color:{color}'><b>{text}</b></span>"

# ---- Google Drive Model Download ----
DETECTION_MODEL_ID = "1IzK4Y-wKDSLjLNUGv2tNDiaRyv0W2Iy6"
CLASSIFICATION_MODEL_ID = "1Ao7o8DekO26M9DcsbwVVwTF3Fwkm7o6S"

@st.cache_resource
def download_models():
    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)
    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"
    # Detection model
    if not os.path.exists(det_path):
        with st.spinner("Loading detection engine..."):
            url = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
            gdown.download(url, det_path, quiet=False)
    # Classification model
    if not os.path.exists(clf_path):
        with st.spinner("Loading classifier..."):
            url = f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}"
            gdown.download(url, clf_path, quiet=False)
    return True, True

det_ready, clf_ready = download_models()

# ---- Model Loading ----
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
        path = "models/classification/best_clf_model.pth"
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Classification model error: {e}")
        return None

@st.cache_resource
def load_detection_model():
    if not det_ready:
        return None
    try:
        path = "models/detection/best.pt"
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Detection model error: {e}")
        return None

clf_model = load_classification_model()
det_model = load_detection_model()

# ---- Grad-CAM ----
def generate_gradcam(model, img_tensor):
    if model is None:
        return None, None
    try:
        device = next(model.parameters()).device
        model.eval()
        extractor = create_feature_extractor(model.model, {"conv_head": "feat"})
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(device)
            out = extractor(img_tensor)
            preds = model(img_tensor)
            pred_class = preds.argmax(dim=1).item()
        feat_map = out["feat"].squeeze().detach().mean(dim=0).cpu().numpy()
        heatmap = cv2.resize(feat_map, (512, 512))
        heatmap = np.clip(heatmap, 0, None)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return heatmap, pred_class
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")
        return None, None

# ---- UI ----
st.markdown(f"<h1 style='color:{ACCENT};'>🩻 CliniScan: AI-powered Chest X-ray Assistant</h1>", unsafe_allow_html=True)
st.write("Quickly evaluate chest X-rays for lung abnormalities and see explainable AI results.")

st.sidebar.header("About Project")
st.sidebar.markdown("""
**What CliniScan Detects (14 types):**  
- Aortic enlargement  
- Atelectasis  
- Calcification  
- Cardiomegaly  
- Consolidation  
- ILD / Infiltration / Lung Opacity / Nodule/Mass  
- Pleural effusion / thickening / pneumothorax  
- Pulmonary fibrosis  
**Main Output Classes:**  
- Abnormal  
- Normal  
----
*Educational/research demo only.*
""")

st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True, caption="Chest X-ray (input)")

    if clf_model is None or det_model is None:
        st.error("❌ Critical: Model files missing or not loaded. Please check.")
        st.stop()

    col_pred, col_det = st.columns(2, gap="large")

    with col_pred:
        st.markdown(f"<h3 style='color:{ACCENT};'>🔍 Lung Class Diagnosis</h3>", unsafe_allow_html=True)
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
        col_txt = ABNORMAL_C if pred_class == 0 else NORMAL_C
        st.markdown(f"Prediction: {colored_text(class_names[pred_class], col_txt)}", unsafe_allow_html=True)
        st.write(f"Confidence: {probs[0][pred_class]:.2%}")
        st.write("Class probabilities:")
        for i, name in enumerate(class_names):
            st.write(f"{name}: {probs[0][i].item():.2%}")
            st.progress(float(probs[0][i].item()))

        st.subheader("🧠 Grad-CAM Visual Explanation")
        st.write("Highlighted areas (red/yellow) influenced model's decision.")
        heatmap, _ = generate_gradcam(clf_model, img_tensor)
        if heatmap is not None:
            heatmap_cmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_cmap = cv2.cvtColor(heatmap_cmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(np.array(image.resize((512, 512))), 0.7, heatmap_cmap, 0.3, 0)
            st.image(overlay, caption="Grad-CAM focus", use_column_width=True)
        else:
            st.warning("Grad-CAM not available for this image.")

        st.caption("Results for demonstration only. Not suitable for clinical use.")

    with col_det:
        st.markdown(f"<h3 style='color:{ACCENT};'>📦 Abnormality Detection</h3>", unsafe_allow_html=True)
        with st.spinner("Scanning image for regional abnormalities..."):
            results = det_model.predict(np.array(image), conf=0.25, verbose=False)
        res_img = results[0].plot()
        st.image(res_img, caption="Detected regions (Bounding Boxes)", use_column_width=True)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.write("Findings:")
            for i in range(min(5, len(boxes))):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                st.write(f"{i+1}. {det_model.names[cls_id]} ({conf:.2%} confidence)")
                st.progress(conf)
            st.write(f"Total findings: {len(boxes)} | Avg confidence: {float(boxes.conf.mean()):.2%}")
        else:
            st.success("No significant abnormalities detected. Appears normal per AI.")

st.markdown("---")
st.markdown(
    f"""
    <div style='color: gray; font-size: 16px; text-align: center;'>
    <strong>⚠️ Disclaimer:</strong>  
    This is a research/educational prototype.  
    Never use for clinical diagnostics—consult a medical expert for interpretations.<br>
    <hr>
    <a href='https://github.com/Y-SHERISHA/CliniScan-Lung-Anormality-Detection' target='_blank'>GitHub Project</a>
    </div>
    """, unsafe_allow_html=True
)
