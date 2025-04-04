import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from grad_cam import GradCAM, apply_gradcam

# Set page configuration
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🔬",
    layout="centered"
)

# Define the class names for the 23 DermNet classes
class_names = [
    "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis Impetigo",
    "Eczema",
    "Exanthems",
    "Fungal Infections",
    "Hair Loss",
    "Herpes HPV",
    "Light Diseases",
    "Lupus",
    "Melanoma Skin Cancer",
    "Nail Fungus",
    "Poison Ivy",
    "Psoriasis",
    "Scabies Lice",
    "Seborrheic Keratoses",
    "Systemic Disease",
    "Tinea Ringworm",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis"
]

def load_model(model_path):
    """Load the trained ResNet18 model"""
    model_path = "resnet18_image_classification_dermnet.pth"    
    # Create a new instance
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 23)

    # Load the weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()
    return model

def predict(image, model):
    """Process the image and make a prediction"""
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Transform the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return predicted.item(), probabilities[0]

def generate_gradcam(image, model, class_idx):
    """Generate Grad-CAM visualization for the given image and class"""
    # Get the last convolutional layer for Grad-CAM
    target_layer = model.layer4[-1]
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Convert PIL image to RGB numpy array
    image_np = np.array(image)
    
    # Define transform for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image for model input
    input_tensor = transform(image).unsqueeze(0)
    
    # Get Grad-CAM heatmap
    cam = grad_cam(input_tensor, class_idx)
    
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    
    # Normalize the heatmap for better visualization
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (
        heatmap_resized.max() - heatmap_resized.min() + 1e-8)  # Adding small epsilon to avoid division by zero
    
    # Convert the heatmap to a color map
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    # Overlay the heatmap on the original image
    overlayed_image = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
    
    return heatmap_colored, overlayed_image

def main():
    st.title("Skin Disease Classification")
    st.write("Upload an image of a skin condition to get a diagnosis prediction using a ResNet18 model trained on the DermNet dataset.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])
    
    # Add model path input (you would typically hardcode this in production)
    model_path = st.text_input("Enter the path to your trained model:", "model_resnet18_dermnet.pth")
    
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add a prediction button
            if st.button("Diagnose"):
                with st.spinner("Analyzing the image..."):
                    try:
                        # Load model
                        model = load_model(model_path)
                        
                        # Make prediction
                        class_idx, probabilities = predict(image, model)
                        
                        # Display prediction
                        st.success(f"Diagnosis: **{class_names[class_idx]}**")
                        
                        # Display the top 5 probabilities
                        st.subheader("Top 5 Predictions:")
                        probs_percent = probabilities.numpy() * 100
                        top5_probs, top5_indices = torch.topk(probabilities, 5)
                        
                        for i in range(5):
                            idx = top5_indices[i].item()
                            prob = top5_probs[i].item() * 100
                            st.write(f"{i+1}. {class_names[idx]}: {prob:.2f}%")
                            # Create a progress bar for visualization
                            st.progress(float(prob/100))
                        
                        # Generate and display Grad-CAM visualization
                        st.subheader("Grad-CAM Visualization")
                        st.write("Highlighting regions that influenced the prediction:")
                        
                        try:
                            heatmap, overlayed = generate_gradcam(image, model, class_idx)
                            
                            # Display visualizations side by side
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(heatmap, caption="Heatmap", use_column_width=True)
                            with col2:
                                st.image(overlayed, caption="Overlayed Explanation", use_column_width=True)
                                
                            st.info("The highlighted areas show regions that the model focused on when making its prediction.")
                        
                        except Exception as e:
                            st.error(f"Error generating Grad-CAM visualization: {e}")
                        
                        # Disclaimer
                        st.warning("This is an AI-assisted diagnosis and should not replace professional medical advice. Please consult a dermatologist for proper diagnosis.")
                    
                    except Exception as e:
                        st.error(f"Error loading model or making prediction: {e}")
                        st.info("Make sure the model path is correct and the model is compatible with the code.")
        
        except Exception as e:
            st.error(f"Error processing the image: {e}")

if __name__ == "__main__":
    main()