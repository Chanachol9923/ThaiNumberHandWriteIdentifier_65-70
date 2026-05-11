from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
import base64, os, time, joblib, io, subprocess, shutil
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass

app = FastAPI()

#      $$$$$$$\  $$\                                         
#      $$  __$$\ \__|                                        
#      $$ |  $$ |$$\ $$$$$$$\   $$$$$$\  $$\   $$\  $$$$$$$\ 
#      $$$$$$$\ |$$ |$$  __$$\ $$  __$$\ $$ |  $$ |$$  _____|
#      $$  __$$\ $$ |$$ |  $$ |$$ /  $$ |$$ |  $$ |\$$$$$$\  
#      $$ |  $$ |$$ |$$ |  $$ |$$ |  $$ |$$ |  $$ | \____$$\ 
#      $$$$$$$  |$$ |$$ |  $$ |\$$$$$$$ |\$$$$$$  |$$$$$$$  |
#      \_______/ \__|\__|  \__| \____$$ | \______/ \_______/ 
#                              $$\   $$ |                    
#                              \$$$$$$  |                    
#                               \______/                     


#1) ชณชล พลเขตรกิจ 1660705755
#2) เอกราช สุ่มมาตย์ 1660707512
#3) วรรณพงษ์ ปกครอง 1660705888
#4) ยุธนากร โพธิ์อยู่ 1660705979
#5) ณภัทร พินิจทรัพย์ 1660700475

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DEFAULT_MODEL_NAME = "thai_BingusHandWrite_Graduated_v2.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

current_model = None
active_model_name = DEFAULT_MODEL_NAME


def preprocess_image(img: Image.Image):
    arr = np.array(img)
    if img.mode == 'RGBA' and arr[:, :, :3].max() == 0 and arr[:, :, 3].max() > 0:
        img = Image.fromarray(arr[:, :, 3], mode='L')
    else:
        img = img.convert('L')
    if np.array(img).mean() > 128:
        img = ImageOps.invert(img)
    img = img.point(lambda p: 255 if p > 30 else 0)
    bbox = img.getbbox()
    if bbox is None: return None
    img = img.crop(bbox)
    arr_crop = np.array(img)
    if arr_crop.sum() > 0:
        cy, cx = center_of_mass(arr_crop)
        shift_y, shift_x = int(arr_crop.shape[0]/2 - cy), int(arr_crop.shape[1]/2 - cx)
        img = img.transform(img.size, Image.AFFINE, (1, 0, -shift_x, 0, 1, -shift_y), fillcolor=0)
    return img.resize((28, 28), Image.Resampling.LANCZOS)

def extract_features(img_array: np.ndarray) -> np.ndarray:
    img = img_array.reshape(28, 28)
    pixels = img.flatten() / 255.0
    zone_features = []
    for r in range(4):
        for c in range(4):
            zone = img[r*7:(r+1)*7, c*7:(c+1)*7]
            zone_features.append(zone.mean() / 255.0)
            zone_features.append(zone.std() / 255.0)
    h_proj = img.mean(axis=1) / 255.0
    v_proj = img.mean(axis=0) / 255.0
    diag = np.array([np.trace(img, offset=k) for k in range(-7, 8)]) / (28 * 255.0)
    return np.concatenate([pixels, zone_features, h_proj, v_proj, diag])

@app.on_event("startup")
async def startup_event():
    global current_model
    if os.path.exists(MODEL_PATH):
        try: current_model = joblib.load(MODEL_PATH)
        except: pass

@app.get("/", response_class=HTMLResponse)
async def read_index():
    path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return f.read()
    return "Error: static/index.html not found."

@app.post("/predict")
async def predict(image: str = Form(...)):
    if current_model is None: return {"prediction": "No Model", "conf": 0}
    try:
        header, encoded = image.split(",", 1)
        data = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(data))
        img_clean = preprocess_image(img)
        if img_clean is None: return {"prediction": "Empty", "conf": 0}
        features = extract_features(np.array(img_clean)).reshape(1, -1)
        prediction = current_model.predict(features)[0]
        mapping = {"65":"๖๕", "66":"๖๖", "67":"๖๗", "68":"๖๘", "69":"๖๙", "70":"๗๐"}
        conf = round(float(np.max(current_model.predict_proba(features))) * 100, 2)
        return {"prediction": mapping.get(str(prediction), str(prediction)), "conf": conf, "model": active_model_name}
    except: return {"prediction": "Error", "conf": 0}

@app.get("/list-models")
async def list_models():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    return {"models": sorted(files), "active": active_model_name}


@app.get("/check-active-model")
async def check_active_model():
    return {
        "active_in_ram": active_model_name,
        "is_loaded": current_model is not None
    }

@app.post("/select-model")
async def select_model(model_name: str = Form(...)):
    global current_model, active_model_name
    path = os.path.join(MODEL_DIR, model_name)
    try:
 
        new_load = joblib.load(path)
        current_model = new_load
        active_model_name = model_name
        print(f"DEBUG: System switched to {model_name}")
        return {"status": "success", "message": f"Activated: {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train-stream")
async def train_stream():
    def run_training():
        train_script = os.path.join(BASE_DIR, "BingusSchoolForTraining.py")
        process = subprocess.Popen(
            ["python", "-u", train_script], 
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace"
        )
        for line in iter(process.stdout.readline, ""):
            yield f"data: {line}\n\n"
        process.stdout.close()
        process.wait()
        global current_model
        if os.path.exists(MODEL_PATH): current_model = joblib.load(MODEL_PATH)
        yield "data: --- FINISHED ---\n\n"
    return StreamingResponse(run_training(), media_type="text/event-stream")

@app.post("/update-model")
async def update_model(file: UploadFile = File(...)):
    global current_model, active_model_name
    path = os.path.join(MODEL_DIR, file.filename)
    content = await file.read()
    with open(path, "wb") as f: f.write(content)
    current_model = joblib.load(path)
    active_model_name = file.filename
    return {"status": "Model uploaded and activated!"}

@app.get("/download-model")
async def download_model():
    path = os.path.join(MODEL_DIR, active_model_name)
    if os.path.exists(path):
        return FileResponse(path=path, filename=active_model_name)
    raise HTTPException(status_code=404, detail="Model file not found.")

@app.get("/download-dataset")
async def download_dataset():
    """ZIP และส่งไฟล์ Dataset ทั้งหมดให้ User"""
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        raise HTTPException(status_code=404, detail="No dataset found to download.")
    
    zip_base_name = os.path.join(BASE_DIR, "bingus_dataset")

    shutil.make_archive(zip_base_name, 'zip', DATASET_DIR)
    return FileResponse(path=f"{zip_base_name}.zip", filename="bingus_dataset.zip")

@app.get("/get-count/{label}")
async def get_count(label: str):
    folder = os.path.join(DATASET_DIR, label)
    count = len([f for f in os.listdir(folder) if f.lower().endswith('.png')]) if os.path.exists(folder) else 0
    return {"count": count}

@app.post("/save-sample")
async def save_sample(label: str = Form(...), image: str = Form(...)):
    folder = os.path.join(DATASET_DIR, label)
    os.makedirs(folder, exist_ok=True)
    header, encoded = image.split(",", 1)
    with open(os.path.join(folder, f"{int(time.time() * 1000)}.png"), "wb") as f:
        f.write(base64.b64decode(encoded))
    return {"status": "success"}