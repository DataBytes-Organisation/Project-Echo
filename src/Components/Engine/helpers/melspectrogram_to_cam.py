import numpy as np
import cv2
import torch
from torchvision import models, transforms

def convert(image):
  # Load the pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Hook to extract features from the last convolutional layer (layer4)
    class SaveFeatures:
        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_fn)
        def hook_fn(self, module, input, output):
            self.features = output.data.numpy()
        def remove(self):
            self.hook.remove()

    final_layer = model._modules.get('layer4')  # Use layer4 to match the weight_softmax dimensions
    activated_features = SaveFeatures(final_layer)

    # Dummy forward pass with a random tensor to trigger the hook
    dummy_input = torch.ones((1, 3, 224, 224))
    with torch.no_grad():
        output = model(dummy_input)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        pred_class = probabilities.argmax().item()

    activated_features.remove()

    # Generate CAM using the matching layer
    def get_cam(features, weight_softmax, class_idx):
        nc, h, w = features.shape[1:]
        cam = weight_softmax[class_idx].dot(features.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return cam_img

    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].data.numpy())

    cam = get_cam(activated_features.features, weight_softmax, pred_class)

    # Print the shape of the original CAM
    print("Original CAM shape:", cam.shape)

    # Resize CAM to match the input image (Mel spectrogram) dimensions using INTER_CUBIC
    input_height, input_width = image.shape[1], image.shape[2]
    cam_resized = cv2.resize(cam, (input_width, input_height), interpolation=cv2.INTER_CUBIC)

    # Print the shape of the resized CAM
    print("Resized CAM shape:", cam_resized.shape)

    # Apply heatmap to the CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # Convert image (Mel spectrogram) from tensor to numpy array
    image_np = image.numpy()[0]
    image_np = np.uint8(255 * image_np)

    # Print the shape of the Mel spectrogram image
    print("Mel spectrogram shape:", image_np.shape)

    # Apply Gaussian blur to the heatmap to smooth the CAM
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Overlay CAM on Mel spectrogram
    alpha = 0.6  # Adjust this value to balance the overlay
    overlayed_image = cv2.addWeighted(image_np, alpha, heatmap, 1 - alpha, 0)

    return overlayed_image


def get_cam(features, weight_softmax, class_idx):
    # features.shape is (batch_size, channels, height, width)
    nc, h, w = features.shape[1:]
    
    # Flatten the spatial dimensions (h, w) for each channel
    cam = np.dot(weight_softmax[class_idx], features.reshape(nc, h * w))
    
    # Reshape the cam back to the spatial dimensions
    cam = cam.reshape(h, w)
    
    # Normalize the CAM to [0, 1]
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    
    return cam