import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List all required packages here:
required_packages = [
    "albumentations",
    "torch",
    "torchvision",
    "timm",
    "streamlit",
    "pillow",
    "pytorch-grad-cam"
]

for pkg in required_packages:
    try:
        __import__(pkg if pkg != "pytorch-grad-cam" else "pytorch_grad_cam")
    except ImportError:
        install(pkg)
# app.py - Streamlit Deployment for CliniScan Chest X-ray Classification

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os

st.set_page_config(page_title="CliniScan Chest X-ray Classification", layout="wide")

st.title("CliniScan Chest X-ray Classifier & Grad-CAM Visualizer")
st.write("Upload a chest X-ray and get instant model prediction with interpretability.")

# ---- Model + Transform Definitions ----

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=False,
                                      num_classes=num_classes, drop_rate=dropout)
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetClassifier(num_classes=2, dropout=0.3).to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, device

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def preprocess_image(file):
    image = np.array(Image.open(file).convert("RGB"))
    image_resized = cv2.resize(image, (512, 512))
    image_norm = image_resized.astype(np.float32) / 255.0
    augmented = transform(image=image_resized)
    input_tensor = augmented['image'].unsqueeze(0)
    return image, image_resized, image_norm, input_tensor

def run_inference(model, device, input_tensor):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probs[0, predicted_class].item()
    return predicted_class, confidence, probs

def generate_gradcam(model, device, input_tensor, image_norm, predicted_class):
    target_layers = [model.model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type == 'cuda'))
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_class)])[0, :]
    cam_image = show_cam_on_image(image_norm, grayscale_cam, use_rgb=True)
    return cam_image

# ---- Main App Logic ----

with st.sidebar:
    st.header("Model & Configuration")
    weights_path = st.text_input("Best model path", "cliniscan_classification_95.20pct.pth")
    if st.button("Show file info"):
        st.write(f"Exists: {os.path.exists(weights_path)}")

model, device = load_model(weights_path)

uploaded_file = st.file_uploader("Upload a chest X-ray image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.success("Image received! Running prediction...")
    image, image_resized, image_norm, input_tensor = preprocess_image(uploaded_file)
    predicted_class, confidence, probs = run_inference(model, device, input_tensor)
    result_label = "Normal" if predicted_class == 1 else "Abnormal"
    result_color = "green" if predicted_class == 1 else "red"

    st.subheader("Results")
    st.markdown(f"### Prediction: <span style='color:{result_color}'>{result_label}</span>", unsafe_allow_html=True)
    st.write(f"Confidence: **{confidence*100:.2f}%**")
    st.write(f"Probability Breakdown: Abnormal: **{probs[0,0].item()*100:.2f}%**, Normal: **{probs[0,1].item()*100:.2f}%**")

    st.image(image, caption="Original Uploaded Image", width=300)
    cam_image = generate_gradcam(model, device, input_tensor, image_norm, predicted_class)
    st.image(cam_image, caption="Grad-CAM Heatmap Overlay", width=300)

    if predicted_class == 0:
        if confidence > 0.9:
            st.warning("High confidence abnormality detected. Recommend specialist review.")
        elif confidence > 0.75:
            st.warning("Probable abnormality detected. Further investigation advised.")
        else:
            st.info("Possible abnormality detected. Consider additional imaging.")
    else:
        if confidence > 0.9:
            st.success("High confidence normal. No immediate concerns.")
        elif confidence > 0.75:
            st.info("Likely normal. Routine follow-up recommended.")
        else:
            st.warning("Uncertain classification. Consider clinical correlation.")

    st.caption(f"Model accuracy on validation: {checkpoint['acc']:.2f}%")

else:
    st.info("Please upload a chest X-ray image to start the analysis.")

st.write("---")
st.markdown("Built with :blue[Streamlit] and :red[PyTorch]. For more information, contact your project lead.")
