import cv2
import numpy as np
import os

class KNN_Model:
    def __init__(self):
        self.model = cv2.ml.KNearest_create()
        self.trained = False

    def load_data_and_train(self, class_file="classifications.txt", img_file="flattened_images.txt"):
        if not os.path.exists(class_file) or not os.path.exists(img_file):
            print("Lỗi: Không tìm thấy dữ liệu train.")
            return False

        try:
            npaClassifications = np.loadtxt(class_file, np.float32)
            npaFlattenedImages = np.loadtxt(img_file, np.float32)
            npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
            
            self.model.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
            self.trained = True
            print(">>> Model KNN: Huấn luyện thành công.")
            return True
        except Exception as e:
            print(f"Lỗi train model: {e}")
            return False

    def predict(self, roi_img):
        if not self.trained: return "?"
        
        # Resize về 20x30 giống lúc train
        roi_small = cv2.resize(roi_img, (20, 30))
        roi_flat = roi_small.reshape((1, 20 * 30)).astype(np.float32)
        
        # Dự đoán (K=3)
        retval, results, neigh_resp, dists = self.model.findNearest(roi_flat, k=3)

        return str(chr(int(results[0][0])))
