import cv2
import matplotlib.pyplot as plt
import preprocess
import detection
import segmentation
import model_knn

def main():
    # --- CẤU HÌNH ---
    IMAGE_PATH = './LP-characters/images/0010.png' 

    # 1. Load Model
    knn = model_knn.KNN_Model()
    knn.load_data_and_train()

    original_img = preprocess.load_image(IMAGE_PATH)
    if original_img is None: return

    # ========================================================
    # GIAI ĐOẠN 1: DETECTION & HIỂN THỊ CHI TIẾT
    # ========================================================
    plate_img, debug_step1 = detection.detect_plate(original_img)
    
    # --- VẼ HÌNH BƯỚC 1 (Giống Notebook) ---
    fig1, ax1 = plt.subplots(2, 3, figsize=(15, 8))
    fig1.suptitle("BƯỚC 1: PHÁT HIỆN BIỂN SỐ", fontsize=16)

    # Ảnh 1: Ảnh gốc + Contour tìm được
    debug_img_draw = original_img.copy()
    if debug_step1["contour"] is not None:
        cv2.drawContours(debug_img_draw, [debug_step1["contour"]], -1, (0, 255, 0), 2)
    
    ax1[0, 0].imshow(cv2.cvtColor(debug_img_draw, cv2.COLOR_BGR2RGB))
    ax1[0, 0].set_title(f"1. Phát hiện: {debug_step1['method']}")

    ax1[0, 1].imshow(debug_step1["gray"], cmap='gray'); ax1[0, 1].set_title("2. Ảnh Xám")
    ax1[0, 2].imshow(debug_step1["blur"], cmap='gray'); ax1[0, 2].set_title("3. Blur")
    ax1[1, 0].imshow(debug_step1["binary"], cmap='gray'); ax1[1, 0].set_title("4. Nhị phân (Otsu)")
    ax1[1, 1].imshow(debug_step1["edge"], cmap='gray'); ax1[1, 1].set_title("5. Cạnh (Edges)")

    if plate_img is not None:
        ax1[1, 2].imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)); ax1[1, 2].set_title("6. Kết quả Cắt")
    else:
        ax1[1, 2].text(0.5, 0.5, "Failed", ha='center'); ax1[1, 2].set_title("6. Thất bại")
        print("Không tìm thấy biển số. Dừng chương trình hoặc dùng Fallback.")
        # Nếu muốn dùng Fallback ảnh gốc thì uncomment dòng dưới:
        # plate_img = original_img 
        plt.show()
        return

    for row in ax1:
        for col in row: col.axis('off')
    plt.tight_layout()
    plt.show() # Hiện bảng 1

    # ========================================================
    # GIAI ĐOẠN 2: SEGMENTATION & RECOGNITION
    # ========================================================
    char_objs, plate_resized, binary_char = segmentation.find_characters(plate_img)
    
    # Nhận diện
    final_text = ""
    for char_obj in char_objs:
        roi = char_obj["roi"]
        predicted_char = knn.predict(roi)
        final_text += predicted_char
        char_obj["char"] = predicted_char

    print(f"\n>>> KẾT QUẢ: {final_text}")

    # --- VẼ HÌNH BƯỚC 2 (Chi tiết phân đoạn) ---
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f"BƯỚC 2: NHẬN DIỆN - KẾT QUẢ: {final_text}", fontsize=16)

    # Ảnh 1: Biển số sau resize
    if plate_resized is not None:
        ax2[0].imshow(cv2.cvtColor(plate_resized, cv2.COLOR_BGR2RGB))
        ax2[0].set_title("1. Biển số (Resize 60px)")
    
    # Ảnh 2: Nhị phân tách chữ
    if binary_char is not None:
        ax2[1].imshow(binary_char, cmap='gray')
        ax2[1].set_title("2. Phân đoạn (Adaptive Threshold)")

    # Ảnh 3: Kết quả vẽ khung
    result_draw = plate_resized.copy() if plate_resized is not None else None
    if result_draw is not None:
        for char_obj in char_objs:
            (x, y, w, h) = char_obj["rect"]
            text = char_obj["char"]
            cv2.rectangle(result_draw, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(result_draw, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        ax2[2].imshow(cv2.cvtColor(result_draw, cv2.COLOR_BGR2RGB))
        ax2[2].set_title(f"3. Nhận diện: {len(char_objs)} ký tự")

    for a in ax2: a.axis('off')
    plt.tight_layout()
    plt.show() # Hiện bảng 2

if __name__ == "__main__":
    main()