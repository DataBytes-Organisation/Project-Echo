import numpy as np
import cv2
import torch
from torchvision import models, transforms

# def convert(image):
#     # Load a pre-trained ResNet model from torchvision and set it to evaluation mode
#     model = models.resnet18(pretrained=True)
#     model.eval()

#     # Ensure Mel spectrogram is a 3-channel image
#     if len(mel_spectrogram.shape) == 2:  # If it's a 2D array, convert to 3 channels
#         mel_spectrogram = np.stack([mel_spectrogram] * 3, axis=-1)

#     # Resize the Mel spectrogram to match the expected input size for ResNet (224x224)
#     resized_mel_spectrogram = cv2.resize(mel_spectrogram, (224, 224))

#     # Convert the resized Mel spectrogram to a PyTorch tensor and normalize
#     preprocess = transforms.Compose([
#         transforms.ToTensor(),  # Convert image to a PyTorch tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
#     ])
#     input_tensor = preprocess(resized_mel_spectrogram).unsqueeze(0)  # Add a batch dimension

#     # Save the features of the last convolutional layer for generating CAM
#     class SaveFeatures:
#         def __init__(self, module):
#             self.hook = module.register_forward_hook(self.hook_fn)
#         def hook_fn(self, module, input, output):
#             self.features = output.data.numpy()  # Save the output features as a numpy array
#         def remove(self):
#             self.hook.remove()  # Remove the hook when done

#     # Hook to the final convolutional layer of ResNet
#     final_layer = model._modules.get('layer4')
#     activated_features = SaveFeatures(final_layer)

#     # Perform a forward pass through the model to get the predictions
#     with torch.no_grad():
#         output = model(input_tensor)  # Get model output
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)

#     # Get the predicted class index (the one with the highest probability)
#     pred_class = probabilities.argmax().item()
#     print(f'Predicted class: {pred_class}')

#     # Remove the hook after getting the features
#     activated_features.remove()

#     # Generate CAM
#     def get_cam(feature_conv, weight_fc, class_idx):
#         _, nc, h, w = feature_conv.shape
#         cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
#         cam = cam.reshape(h, w)
#         cam = cam - np.min(cam)
#         cam_img = cam / np.max(cam)
#         return cam_img

#     weight_softmax_params = list(model._modules.get('fc').parameters())
#     weight_softmax = np.squeeze(weight_softmax_params[0].data.numpy())

#     cam = get_cam(activated_features.features, weight_softmax, pred_class)

#     # Resize CAM to match the original Mel spectrogram dimensions
#     input_height, input_width = mel_spectrogram.shape[:2]
#     cam_resized = cv2.resize(cam, (input_width, input_height))

#     # Apply heatmap
#     heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

#     # Convert Mel spectrogram to uint8 type
#     mel_spectrogram_uint8 = np.uint8(255 * mel_spectrogram / np.max(mel_spectrogram))

#     # Overlay CAM on Mel spectrogram
#     alpha = 0.5  # Transparency factor
#     overlayed_image = cv2.addWeighted(mel_spectrogram_uint8, alpha, heatmap, 1 - alpha, 0)

#     # Draw bounding box
#     # Threshold CAM to find the region with significant activations
#     threshold = np.max(cam_resized) * 0.5  # Adjust threshold as needed
#     mask = cam_resized > threshold
#     y_indices, x_indices = np.where(mask)
#     x_min, x_max = np.min(x_indices), np.max(x_indices)
#     y_min, y_max = np.min(y_indices), np.max(y_indices)
#     cv2.rectangle(overlayed_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#     # Display the result
#     plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
#     plt.title(f'Predicted Class: {pred_class}')
#     plt.axis('off')
#     plt.show()

import numpy as np
import cv2
import torch
from torchvision import models, transforms

def convert(image):
    # Load a pre-trained ResNet model from torchvision and set it to evaluation mode
    model = models.resnet18(pretrained=True)
    model.eval()

    # Ensure Mel spectrogram is a 3-channel image
    if len(image.shape) == 2:  # If it's a 2D array, convert to 3 channels
        image = np.stack([image] * 3, axis=-1)

    # Resize the Mel spectrogram to match the expected input size for ResNet (224x224)
    resized_image = cv2.resize(image, (224, 224))

    # Convert the resized image to a PyTorch tensor and normalize
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    input_tensor = preprocess(resized_image).unsqueeze(0)  # Add a batch dimension

    # Save the features of the last convolutional layer for generating CAM
    class SaveFeatures:
        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_fn)
        def hook_fn(self, module, input, output):
            self.features = output.data.numpy()  # Save the output features as a numpy array
        def remove(self):
            self.hook.remove()  # Remove the hook when done

    # Hook to the final convolutional layer of ResNet
    final_layer = model._modules.get('layer4')
    activated_features = SaveFeatures(final_layer)

    # Perform a forward pass through the model to get the predictions
    with torch.no_grad():
        output = model(input_tensor)  # Get model output
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the predicted class index (the one with the highest probability)
    pred_class = probabilities.argmax().item()
    print(f'Predicted class: {pred_class}')

    # Remove the hook after getting the features
    activated_features.remove()

    # Generate CAM
    def get_cam(feature_conv, weight_fc, class_idx):
        _, nc, h, w = feature_conv.shape
        cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        return cam_img

    weight_softmax_params = list(model._modules.get('fc').parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].data.numpy())

    cam = get_cam(activated_features.features, weight_softmax, pred_class)

    # Resize CAM to match the original image dimensions
    input_height, input_width = image.shape[:2]
    cam_resized = cv2.resize(cam, (input_width, input_height))

    # Apply heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # Convert image to uint8 type
    image_uint8 = np.uint8(255 * image / np.max(image))

    # Overlay CAM on the image
    alpha = 0.5  # Transparency factor
    overlayed_image = cv2.addWeighted(image_uint8, alpha, heatmap, 1 - alpha, 0)

    # Draw bounding box
    # Threshold CAM to find the region with significant activations
    threshold = np.max(cam_resized) * 0.5  # Adjust threshold as needed
    mask = cam_resized > threshold
    y_indices, x_indices = np.where(mask)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    cv2.rectangle(overlayed_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Return the final processed image
    return overlayed_image
