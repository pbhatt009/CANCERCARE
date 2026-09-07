import io
import json

import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = "model_best.pth"
CLASS_MAP_PATH = "class_mapping.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_class_mapping(mapping_path: str):
    with open(mapping_path, "r") as file_handle:
        class_to_idx = json.load(file_handle)
    return {index: class_name for class_name, index in class_to_idx.items()}


@st.cache_resource
def load_model(model_path: str, num_classes: int):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict_image(image_bytes: bytes, model, idx_to_class):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class_idx = torch.max(probabilities, 1)

    class_idx = top_class_idx.item()
    return idx_to_class[class_idx], round(top_prob.item() * 100, 2)


def genomic_lookup(query_type: str, query: str):
    if query_type == "gene":
        url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{query}?content-type=application/json"
    else:
        url = f"https://rest.ensembl.org/variation/human/{query}?content-type=application/json"

    response = requests.get(url, timeout=20)
    if response.status_code == 200:
        return response.json(), None
    return None, f"{query_type.capitalize()} '{query}' not found."


st.set_page_config(page_title="CancerCare Dashboard", page_icon="🩺", layout="wide")

st.title("CancerCare Dashboard")
st.caption("Medical image classification and genomic lookup.")

try:
    idx_to_class = load_class_mapping(CLASS_MAP_PATH)
    model = load_model(MODEL_PATH, len(idx_to_class))
    model_ready = True
except Exception as error:
    model_ready = False
    model_error = str(error)

tab_predict, tab_genomic = st.tabs(["Medical Image Classifier", "Genomic Request"])

with tab_predict:
    st.subheader("Upload a medical image")
    if not model_ready:
        st.error(f"Model unavailable: {model_error}")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        preview_columns = st.columns([1, 1])
        with preview_columns[0]:
            st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

        if st.button("Run classification", type="primary", disabled=not model_ready):
            prediction, confidence = predict_image(uploaded_file.getvalue(), model, idx_to_class)
            with preview_columns[1]:
                st.metric("Predicted class", prediction)
                st.metric("Confidence", f"{confidence:.2f}%")
                st.progress(min(max(confidence / 100.0, 0.0), 1.0))

with tab_genomic:
    st.subheader("Genomic lookup")
    query_type = st.selectbox("Query type", ["gene", "variant"])
    query = st.text_input("Enter a gene symbol or variant rsID")

    if st.button("Search genomic data", type="primary"):
        if not query.strip():
            st.warning("Enter a gene symbol or variant rsID first.")
        else:
            result, error = genomic_lookup(query_type, query.strip())
            if error:
                st.error(error)
            else:
                st.success("Lookup complete")
                st.json(result)
