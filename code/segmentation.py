import cv2
import numpy as np
import preprocess

def find_characters(plate_img):
    if plate_img is None: 
        return [], None, None

    # 1. Resize chuẩn chiều cao 60px
    h_plate, w_plate = plate_img.shape[:2]
    new_h = 60
    new_w = int(new_h * (w_plate / h_plate))
    plate_resized = cv2.resize(plate_img, (new_w, new_h))

    # 2. Tiền xử lý tách chữ
    gray = preprocess.to_gray(plate_resized)
    binary = preprocess.make_binary_adaptive(gray)

    # 3. Tìm Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        height_ratio = h / float(new_h)
        area = w * h

        if 0.2 < height_ratio < 0.95 and 0.1 < aspect_ratio < 2.0 and area > 50:
            candidates.append((x, y, w, h))

    if not candidates:
        return [], plate_resized, binary

    # 4. Lọc nhiễu Median Height
    heights = [cand[3] for cand in candidates]
    median_h = np.median(heights)
    
    valid_chars = []
    for (x, y, w, h) in candidates:
        if h > median_h * 0.6: 
            roi = binary[y:y+h, x:x+w]
            valid_chars.append({
                "roi": roi,
                "rect": (x, y, w, h)
            })
            
    valid_chars = sorted(valid_chars, key=lambda k: k["rect"][0])
    
    # Trả về: Danh sách chữ, Ảnh màu đã resize, Ảnh nhị phân (để vẽ hình)
    return valid_chars, plate_resized, binary
