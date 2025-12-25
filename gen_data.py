# File: gen_data.py
import sys
import numpy as np
import cv2
import os

# --- CẤU HÌNH ---
MIN_CONTOUR_AREA = 40
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
IMAGE_PATH = "training_chars.png" # Đảm bảo file ảnh này nằm cùng thư mục

def main():
    # 1. Đọc ảnh training
    if not os.path.exists(IMAGE_PATH):
        print(f"Lỗi: Không tìm thấy file ảnh '{IMAGE_PATH}'")
        return

    imgTrainingNumbers = cv2.imread(IMAGE_PATH)
    if imgTrainingNumbers is None:
        print("Lỗi: Không đọc được dữ liệu ảnh.")
        return

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)

    # 2. Xử lý ảnh (Adaptive Threshold)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)

    cv2.imshow("imgThresh", imgThresh)
    
    # Copy để tìm contour không làm hỏng ảnh gốc
    imgThreshCopy = imgThresh.copy()

    # 3. Tìm Contours
    npaContours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Khởi tạo mảng dữ liệu
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    intClassifications = []

    # Các ký tự hợp lệ (0-9, A-Z)
    intValidChars = [ord(str(i)) for i in range(10)] + [ord(chr(i)) for i in range(65, 91)]

    print("BẮT ĐẦU GÁN NHÃN THỦ CÔNG:")
    print("- Gõ ký tự tương ứng trên bàn phím.")
    print("- Nhấn ESC để thoát.")
    print("-----------------------------------")

    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            # Vẽ khung đỏ bao quanh ký tự đang xét
            cv2.rectangle(imgTrainingNumbers, (intX, intY), (intX+intW,intY+intH), (0, 0, 255), 2)

            # Cắt ảnh ký tự
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            # Hiển thị để người dùng nhìn
            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)
            cv2.imshow("training_numbers.png", imgTrainingNumbers)

            # --- CHỜ NGƯỜI DÙNG GÕ PHÍM ---
            intChar = cv2.waitKey(0)

            if intChar == 27: # Phím ESC
                sys.exit()
            elif intChar in intValidChars:
                # Nếu gõ đúng ký tự hợp lệ -> Lưu lại
                intClassifications.append(intChar)
                
                # Duỗi ảnh thành 1 hàng và thêm vào mảng
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
                
                print(f"Đã lưu: {chr(intChar)}")

    # 4. Lưu kết quả ra file .txt
    fltClassifications = np.array(intClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print("\n\nHOÀN TẤT! Đang lưu dữ liệu...")
    
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)

    print("Đã tạo xong 2 file: classifications.txt và flattened_images.txt")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()