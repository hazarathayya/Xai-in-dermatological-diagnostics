import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Register hooks
        self.hook_handles = []

        # Forward hook to capture activations
        self.hook_handles.append(
            self.target_layer.register_forward_hook(self.save_activation)
        )

        # Backward hook to capture gradients - update this line
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(self.save_gradient)
        )

        self.activations = None

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Grad_output is a tuple, take the first element
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        model_output = self.model(x)

        # Target for backprop
        one_hot = torch.zeros_like(model_output)
        one_hot[0, class_idx] = 1

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        model_output.backward(gradient=one_hot, retain_graph=True)

        # Get weights
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        # Take the channel-wise mean of gradients
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted sum of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to focus on positive influence
        cam = np.maximum(cam, 0)

        return cam



def apply_gradcam(img_path, model, gradcam):
    # Read the image
    image = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Image not found at {img_path}")
        return

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a PIL image
    image_pil = Image.fromarray(image_rgb)

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Apply the transform and add a batch dimension
    input_tensor = transform(image_pil).unsqueeze(0)

    # Get Grad-CAM heatmap (assumes GradCAM class is already defined)
    heatmap = gradcam(input_tensor)  # Grad-CAM returns a heatmap (e.g., as a 2D NumPy array)

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))

    # Normalize the heatmap for better visualization
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

    # Convert the heatmap to a color map
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image (weighted sum)
    overlayed_image = cv2.addWeighted(image_rgb, 0.6, heatmap_colored, 0.4, 0)


    output_path = '/content/gradcam_overlayed_image.jpg'  # Specify your desired path
    overlayed_image_bgr = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV compatibility
    cv2.imwrite(output_path, overlayed_image_bgr)
    print(f"Overlayed Grad-CAM image saved at {output_path}")

    # Plot the result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_colored)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlayed_image)
    plt.title("Overlayed Image")
    plt.axis("off")

    plt.show()

# Apply Grad-CAM to a test image
# apply_gradcam('/content/hives-Urticaria-Acute-78.jpg', model, gradcam)
