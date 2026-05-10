import os, joblib, numpy as np
from PIL import Image, ImageOps
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.ndimage import center_of_mass

# --- [Configuration] ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
# ตรวจสอบว่า Path โมเดลถูกต้อง
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_NAME = os.path.join(MODEL_DIR, "thai_BingusHandWrite_Graduated_v2.pkl")
IMG_SIZE = (28, 28)

def augment_image(img):
    """ทำ Data Augmentation เพื่อเพิ่มความแม่นยำ"""
    images = [img]
    # หมุนภาพ
    for angle in [8, -8, 15, -15]:
        images.append(img.rotate(angle, fillcolor=0))
    # เลื่อนภาพ (Translation)
    for dx, dy in [(2,0), (-2,0), (0,2), (0,-2)]:
        images.append(img.transform(IMG_SIZE, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0))
    # เพิ่ม Noise เล็กน้อย
    arr = np.array(img, dtype=np.float32)
    noisy = np.clip(arr + np.random.normal(0, 10, arr.shape), 0, 255).astype(np.uint8)
    images.append(Image.fromarray(noisy))
    return images

def extract_features(img_array):
    """ดึงข้อมูลคุณลักษณะ (Features) จากรูปภาพ"""
    img = img_array.reshape(28, 28)
    
    # 1. Raw pixels (Normalized)
    pixels = img.flatten() / 255.0
    
    # 2. Zone features (4x4 grid)
    zone_features = []
    zone_size = 7 
    for r in range(4):
        for c in range(4):
            zone = img[r*zone_size:(r+1)*zone_size, c*zone_size:(c+1)*zone_size]
            zone_features.append(zone.mean() / 255.0)
            zone_features.append(zone.std() / 255.0)
    
    # 3. Projections
    h_proj = img.mean(axis=1) / 255.0
    v_proj = img.mean(axis=0) / 255.0
    
    # 4. Diagonal features
    diag = np.array([np.trace(img, offset=k) for k in range(-7, 8)]) / (28 * 255.0)
    
    return np.concatenate([pixels, zone_features, h_proj, v_proj, diag])

def preprocess_image(img):
    """จัดการรูปภาพให้พร้อมสำหรับการดึง Feature (เหมือนใน main.py)"""
    arr = np.array(img)

    # จัดการ Alpha channel จาก Canvas
    if img.mode == 'RGBA' and arr[:,:,:3].max() == 0 and arr[:,:,3].max() > 0:
        img = Image.fromarray(arr[:,:,3], mode='L')
    else:
        img = img.convert('L')

    if np.array(img).mean() > 128:
        img = ImageOps.invert(img)

    # Binary Threshold
    img = img.point(lambda p: 255 if p > 30 else 0)

    # Auto-crop
    bbox = img.getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)

    # จัดตำแหน่งให้อยู่กึ่งกลาง (Center of Mass)
    arr_crop = np.array(img)
    if arr_crop.sum() > 0:
        cy, cx = center_of_mass(arr_crop)
        shift_y = int(arr_crop.shape[0] / 2 - cy)
        shift_x = int(arr_crop.shape[1] / 2 - cx)
        img = img.transform(
            img.size, Image.AFFINE, (1, 0, -shift_x, 0, 1, -shift_y), fillcolor=0
        )

    return img.resize((28, 28), Image.Resampling.LANCZOS)

def load_dataset():
    X, y = [], []
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset folder not found at {DATASET_DIR}")
        return None, None

    # ดึง Label จากชื่อโฟลเดอร์ (65, 66, ...)
    categories = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

    print("--- Starting Data Preparation ---")
    for label in categories:
        label_path = os.path.join(DATASET_DIR, label)
        count = 0
        for img_file in os.listdir(label_path):
            if not img_file.lower().endswith('.png'):
                continue
            try:
                img = Image.open(os.path.join(label_path, img_file))
                img_clean = preprocess_image(img)
                
                if img_clean is None:
                    continue

                aug_images = augment_image(img_clean)
                for a_img in aug_images:
                    features = extract_features(np.array(a_img))
                    X.append(features)
                    y.append(label)
                    count += 1
            except Exception as e:
                continue
        print(f"Label {label}: {count} images processed (including augmentation)")
    return np.array(X), np.array(y)

def train():
    X, y = load_dataset()
    if X is None or len(X) < 10:
        print("Error: Insufficient data for training")
        return

    # แบ่งข้อมูล Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SVM Pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='rbf',
            C=10,
            probability=True,
            random_state=42
        ))
    ])

    print(f"Training model with {len(X_train)} samples...")
    model.fit(X_train, y_train)

    print("\n--- Accuracy Report ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    # บันทึกโมเดล
    os.makedirs(os.path.dirname(MODEL_NAME), exist_ok=True)
    joblib.dump(model, MODEL_NAME)
    print(f"\nModel updated successfully: {MODEL_NAME}")

if __name__ == "__main__":
    train()