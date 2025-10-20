import numpy as np
import torch
import cv2

def create_test_image(width, height, color=(255, 255, 255)):
    """Creates a solid color image."""
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

def create_sharp_image(width, height):
    """Creates a sharp image with a black square on a white background."""
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    start_x = width // 4
    start_y = height // 4
    end_x = start_x + width // 2
    end_y = start_y + height // 2
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

def create_blurry_image(image_tensor, kernel_size=15):
    """Applies Gaussian blur to an image tensor."""
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255
    blurred_image = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
    return torch.from_numpy(blurred_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

def create_low_contrast_image(width, height):
    """Creates a low-contrast image."""
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    image[height//4:height//4*3, width//4:width//4*3, :] = 140
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
