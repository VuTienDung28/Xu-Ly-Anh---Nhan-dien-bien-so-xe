import cv2
import numpy as np

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không đọc được ảnh từ {image_path}")
        return None
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def noise_removal(img):
    # Dùng GaussianBlur như trong Plate_1
    return cv2.GaussianBlur(img, (5, 5), 0)

def make_binary_otsu(img):
    # Dùng cho bước detection
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def make_binary_adaptive(img):
    # Dùng cho bước segmentation (tách chữ)
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)