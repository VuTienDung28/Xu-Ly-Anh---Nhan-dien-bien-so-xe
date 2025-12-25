import cv2
import numpy as np
import preprocess

def find_candidates(edge_img):
    contours, _ = cv2.findContours(edge_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    candidates = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        (center, (w, h), angle) = rect

        if w < h: w, h = h, w; angle += 90
        if angle > 45: angle -= 90
        elif angle < -45: angle += 90
        
        aspect_ratio = w / float(h)
        area = w * h
        
        if aspect_ratio > 1.8 and aspect_ratio < 6.0 and area > 1500:
            candidates.append({
                "contour": c, "rect": rect, "area": area, 
                "angle": angle, "center": center, "size": (w, h)
            })
    return candidates[0] if candidates else None

def crop_plate(img, candidate):
    (w, h) = candidate["size"]
    center = candidate["center"]
    angle = candidate["angle"]

    if abs(angle) < 30:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_full = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        center_homogeneous = np.array([center[0], center[1], 1.0])
        new_center = M.dot(center_homogeneous)
        plate_img = cv2.getRectSubPix(rotated_full, (int(w), int(h)), (new_center[0], new_center[1]))
    else:
        x, y, w_b, h_b = cv2.boundingRect(candidate["contour"])
        plate_img = img[y:y+h_b, x:x+w_b]
    return plate_img

def detect_plate(img):
    # 1. Tiền xử lý
    gray = preprocess.to_gray(img)
    blur = preprocess.noise_removal(gray)
    
    # Tạo ảnh Binary chỉ để hiển thị (Debug)
    binary_display = preprocess.make_binary_otsu(blur) 

    # 2. Tạo 2 loại ảnh cạnh
    edged_canny = cv2.Canny(blur, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    edged_closing = cv2.morphologyEx(edged_canny, cv2.MORPH_CLOSE, kernel)

    # 3. Chạy chiến thuật so sánh 2 vòng
    cand1 = find_candidates(edged_canny)
    cand2 = find_candidates(edged_closing)

    final_cand = None
    method_used = "Thất bại"
    used_edge_img = edged_canny # Mặc định

    if cand1 and cand2:
        if cand2["area"] > cand1["area"] * 1.5:
            final_cand = cand2
            method_used = "Vòng 2 (Closing)"
            used_edge_img = edged_closing
        else:
            final_cand = cand1
            method_used = "Vòng 1 (Canny)"
    elif cand1:
        final_cand = cand1
        method_used = "Vòng 1 (Canny)"
    elif cand2:
        final_cand = cand2
        method_used = "Vòng 2 (Closing)"
        used_edge_img = edged_closing

    plate_img = None
    found_contour = None

    if final_cand:
        plate_img = crop_plate(img, final_cand)
        found_contour = final_cand["contour"]

    # Đóng gói dữ liệu để vẽ hình
    debug_data = {
        "gray": gray,
        "blur": blur,
        "binary": binary_display,
        "edge": used_edge_img,
        "contour": found_contour,
        "method": method_used
    }

    return plate_img, debug_data