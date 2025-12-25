import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# PHẦN 1: CẤU HÌNH & KHỞI TẠO MÔ HÌNH (MODEL)
# =============================================================================

class KNN_Model:
    def __init__(self):
        self.model = cv2.ml.KNearest_create()
        self.trained = False

    def train(self, class_file="classifications.txt", img_file="flattened_images.txt"):
        if not os.path.exists(class_file) or not os.path.exists(img_file):
            print("CẢNH BÁO: Không tìm thấy dữ liệu train (txt). Chạy chế độ không nhận diện.")
            return False

        try:
            npaClassifications = np.loadtxt(class_file, np.float32)
            npaFlattenedImages = np.loadtxt(img_file, np.float32)
            npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
            
            self.model.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
            self.trained = True
            print(">>> [Init] Model KNN đã được huấn luyện xong.")
            return True
        except Exception as e:
            print(f"Lỗi train model: {e}")
            return False

    def predict(self, roi_img):
        if not self.trained: return "?"
        # Resize về 20x30 giống lúc train
        roi_small = cv2.resize(roi_img, (20, 30))
        roi_flat = roi_small.reshape((1, 20 * 30)).astype(np.float32)
        # K=3 để ổn định
        retval, results, neigh_resp, dists = self.model.findNearest(roi_flat, k=3)
        return str(chr(int(results[0][0])))

# =============================================================================
# PHẦN 2: CÁC HÀM XỬ LÝ ẢNH (PREPROCESS)
# =============================================================================

def step1_detect_plate(img):
    """
    Bước 1: Tìm và cắt biển số (Logic 2 vòng: Canny -> Closing)
    Trả về: (plate_img, debug_info_dict)
    """
    # 1. Tiền xử lý
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Chỉ để hiển thị

    # 2. Tạo Edge Map
    edged_canny = cv2.Canny(blur, 30, 150) # Vòng 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    edged_closing = cv2.morphologyEx(edged_canny, cv2.MORPH_CLOSE, kernel) # Vòng 2

    # Hàm tìm ứng viên
    def find_best_candidate(edge_map):
        contours, _ = cv2.findContours(edge_map.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for c in contours:
            rect = cv2.minAreaRect(c)
            (center, (w, h), angle) = rect
            # Chuẩn hóa
            if w < h: w, h = h, w; angle += 90
            if angle > 45: angle -= 90
            elif angle < -45: angle += 90
            
            aspect_ratio = w / float(h)
            area = w * h
            if aspect_ratio > 1.8 and aspect_ratio < 6.0 and area > 1500:
                return {"cnt": c, "rect": rect, "area": area, "angle": angle, "center": center, "size": (w,h)}
        return None

    # 3. Chạy chiến thuật
    cand1 = find_best_candidate(edged_canny)
    cand2 = find_best_candidate(edged_closing)

    final_cand = None
    method_used = "Thất bại"
    used_edge = edged_canny

    if cand1 and cand2:
        if cand2["area"] > cand1["area"] * 1.5:
            final_cand = cand2; method_used = "Vòng 2 (Closing)"; used_edge = edged_closing
        else:
            final_cand = cand1; method_used = "Vòng 1 (Canny)"
    elif cand1:
        final_cand = cand1; method_used = "Vòng 1 (Canny)"
    elif cand2:
        final_cand = cand2; method_used = "Vòng 2 (Closing)"; used_edge = edged_closing

    plate_img = None
    if final_cand:
        (w, h) = final_cand["size"]
        center = final_cand["center"]
        angle = final_cand["angle"]
        if abs(angle) < 30:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_full = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            new_center = M.dot(np.array([center[0], center[1], 1.0]))
            plate_img = cv2.getRectSubPix(rotated_full, (int(w), int(h)), (new_center[0], new_center[1]))
        else:
            x, y, w_b, h_b = cv2.boundingRect(final_cand["cnt"])
            plate_img = img[y:y+h_b, x:x+w_b]

    debug_info = {
        "gray": gray, "blur": blur, "binary": binary_otsu, 
        "edge": used_edge, "method": method_used, 
        "contour": final_cand["cnt"] if final_cand else None
    }
    return plate_img, debug_info

def step2_segment_and_recognize(plate_img, knn_model):
    """
    Bước 2: Tách chữ (Adaptive Threshold + Median Filter) và Nhận diện
    Trả về: (result_chars_list, plate_resized, binary_img)
    """
    if plate_img is None: return [], None, None

    # 1. Resize chuẩn 60px height
    h, w = plate_img.shape[:2]
    new_h = 60
    new_w = int(new_h * (w / h))
    plate_resized = cv2.resize(plate_img, (new_w, new_h))

    # 2. Phân đoạn
    gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 3. Tìm contour chữ
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        asp = cw / float(ch)
        h_ratio = ch / float(new_h)
        area = cw * ch
        # Lọc thô
        if 0.2 < h_ratio < 0.95 and 0.1 < asp < 2.0 and area > 50:
            candidates.append((x, y, cw, ch))

    if not candidates: return [], plate_resized, binary

    # 4. Lọc nhiễu (Median Filter)
    heights = [item[3] for item in candidates]
    median_h = np.median(heights)
    
    final_chars = []
    for (x, y, cw, ch) in candidates:
        if ch > median_h * 0.6: # Cao > 60% trung bình
            roi = binary[y:y+ch, x:x+cw]
            char_text = knn_model.predict(roi)
            final_chars.append({"char": char_text, "rect": (x, y, cw, ch)})
            
    # Sắp xếp trái -> phải
    final_chars = sorted(final_chars, key=lambda k: k["rect"][0])
    
    return final_chars, plate_resized, binary

# =============================================================================
# PHẦN 3: MAIN FLOW (CHẠY CHƯƠNG TRÌNH)
# =============================================================================

def main():
    # --- CẤU HÌNH ---
    IMAGE_PATH = './LP-characters/images/0010.png'  # <--- THAY ĐỔI ẢNH CỦA BẠN Ở ĐÂY
    
    print(f"\n--- Đang xử lý: {IMAGE_PATH} ---")
    if not os.path.exists(IMAGE_PATH):
        print("Lỗi: File ảnh không tồn tại!")
        return

    # 1. Load Model
    knn = KNN_Model()
    knn.train()

    # 2. Đọc ảnh gốc
    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None: return

    # ---------------------------------------------------------
    # GIAI ĐOẠN 1: DETECT
    # ---------------------------------------------------------
    plate_img, debug1 = step1_detect_plate(original_img)
    
    # Hiển thị Bước 1
    fig1, ax1 = plt.subplots(2, 3, figsize=(14, 8))
    fig1.canvas.manager.set_window_title('Bước 1: Cắt biển số')
    
    debug_draw = original_img.copy()
    if debug1["contour"] is not None: cv2.drawContours(debug_draw, [debug1["contour"]], -1, (0,255,0), 2)
    
    ax1[0,0].imshow(cv2.cvtColor(debug_draw, cv2.COLOR_BGR2RGB)); ax1[0,0].set_title(f"Phát hiện: {debug1['method']}")
    ax1[0,1].imshow(debug1["gray"], cmap='gray'); ax1[0,1].set_title("Ảnh Xám")
    ax1[0,2].imshow(debug1["blur"], cmap='gray'); ax1[0,2].set_title("Blur")
    ax1[1,0].imshow(debug1["binary"], cmap='gray'); ax1[1,0].set_title("Nhị phân (Otsu)")
    ax1[1,1].imshow(debug1["edge"], cmap='gray'); ax1[1,1].set_title("Cạnh (Edge)")
    
    if plate_img is not None:
        ax1[1,2].imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)); ax1[1,2].set_title("Kết quả cắt")
    else:
        ax1[1,2].text(0.5, 0.5, "Thất bại", ha='center'); ax1[1,2].set_title("Thất bại")
        print("Không cắt được biển. Dùng Fallback ảnh gốc...")
        plate_img = original_img # Fallback

    for row in ax1:
        for col in row: col.axis('off')
    plt.tight_layout()
    plt.show() # <--- Dừng lại để xem ảnh bước 1

    # ---------------------------------------------------------
    # GIAI ĐOẠN 2: RECOGNIZE
    # ---------------------------------------------------------
    chars_found, plate_resized, binary_char = step2_segment_and_recognize(plate_img, knn)
    
    full_text = "".join([c["char"] for c in chars_found])
    print(f"\n>>> BIỂN SỐ NHẬN DIỆN: {full_text}")

    # Hiển thị Bước 2
    if plate_resized is not None:
        fig2, ax2 = plt.subplots(1, 3, figsize=(14, 5))
        fig2.canvas.manager.set_window_title(f'Bước 2: Kết quả - {full_text}')
        
        ax2[0].imshow(cv2.cvtColor(plate_resized, cv2.COLOR_BGR2RGB)); ax2[0].set_title("Biển số (Resize)")
        ax2[1].imshow(binary_char, cmap='gray'); ax2[1].set_title("Phân đoạn")
        
        res_draw = plate_resized.copy()
        for c in chars_found:
            (x,y,w,h) = c["rect"]
            cv2.rectangle(res_draw, (x,y), (x+w, y+h), (0,255,0), 1)
            cv2.putText(res_draw, c["char"], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            
        ax2[2].imshow(cv2.cvtColor(res_draw, cv2.COLOR_BGR2RGB)); ax2[2].set_title(f"Kết quả: {full_text}")
        
        for a in ax2: a.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
