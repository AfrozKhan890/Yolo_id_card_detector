import os
import uuid
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO

import pytesseract 

# ---------- (Windows only) set Tesseract path if needed ----------
# If Windows & Tesseract not in PATH, uncomment and set the path:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.secret_key = "3UiDUzFW4fXZUqTmzIxAhaOdyA9rr6cF"

OUTPUT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- Load YOLO model ----------
# Place your trained best.pt in project root
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
# print("Model class names:", model.names)

# Helper: file type check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper: OCR pre-processing (improves accuracy)
def preprocess_for_ocr(bgr_img):
    # Convert to grayscale
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Upscale small crops for better OCR
    h, w = gray.shape[:2]
    scale = 2 if max(h, w) < 800 else 1
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Denoise + binarize
    gray = cv2.bilateralFilter(gray, 5, 55, 55)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th

# Helper: Run Tesseract OCR and return cleaned text
def run_ocr_on_crop(bgr_crop):
    proc = preprocess_for_ocr(bgr_crop)
    # You can tweak PSM if needed: 6 (block of text), 7 (single line), 8 (single word)
    config = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(proc, lang="eng", config=config)
    # Clean up whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Only JPG, JPEG, PNG allowed', 'error')
            return redirect(request.url)

        try:
            # Read image in memory (without saving uploads/)
            file_bytes = file.read()
            np_img = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Unique output file for annotated result
            unique_id = str(uuid.uuid4())[:8]
            result_filename = f"result_{unique_id}.jpg"
            result_path = os.path.join(app.config['OUTPUT_FOLDER'], result_filename)

            # ---------- Run YOLO detection ----------
            # Lower conf if needed (0.25~0.35 is common); adjust as per dataset
            results = model.predict(img, conf=0.3, verbose=False)

            # Save visualization (annotated image)
            for result in results:
                im_array = result.plot()
                im = Image.fromarray(im_array[..., ::-1])  # BGR->RGB fix
                im.save(result_path)
                break

            # ---------- Parse detections ----------
            id_detected = False
            detection_confidence = 0.0
            extracted_text = ""

            # Collect boxes by class
            id_card_boxes = []  # for "with_id" or card region
            id_text_boxes = []  # for "id_text" / "text area"

            # Model classes (e.g., ["no_id", "with_id", "id_text"])
            class_names = model.names

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    class_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]
                    conf = float(box.conf)
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy.tolist()

                    # Track "with_id" presence & confidence
                    if class_name.lower() in ["with_id", "id_card", "card", "withid"]:
                        id_detected = id_detected or (conf > 0.3)
                        detection_confidence = max(detection_confidence, conf)
                        id_card_boxes.append((conf, (x1, y1, x2, y2)))

                    # If you have a separate "id_text" class in your dataset
                    if class_name.lower() in ["id_text", "text_area", "text", "card_text"]:
                        id_text_boxes.append((conf, (x1, y1, x2, y2)))

            # ---------- OCR Strategy ----------
            # 1) If id_text boxes exist, OCR each (sorted by conf desc)
            # 2) Else if id_card box exists, OCR the card region
            # 3) Else nothing to OCR

            texts = []

            if id_text_boxes:
                # Sort by confidence high->low, process up to a few boxes
                id_text_boxes.sort(key=lambda t: t[0], reverse=True)
                for conf, (x1, y1, x2, y2) in id_text_boxes[:5]:
                    # Safe padding
                    pad = 4
                    x1p = max(0, x1 - pad)
                    y1p = max(0, y1 - pad)
                    x2p = min(img.shape[1], x2 + pad)
                    y2p = min(img.shape[0], y2 + pad)
                    crop = img[y1p:y2p, x1p:x2p]
                    if crop.size > 0:
                        txt = run_ocr_on_crop(crop)
                        if txt:
                            texts.append(txt)

            elif id_card_boxes:
                # Use the highest-confidence card box
                id_card_boxes.sort(key=lambda t: t[0], reverse=True)
                _, (x1, y1, x2, y2) = id_card_boxes[0]
                # Slight padding
                pad = 6
                x1p = max(0, x1 - pad)
                y1p = max(0, y1 - pad)
                x2p = min(img.shape[1], x2 + pad)
                y2p = min(img.shape[0], y2 + pad)
                crop = img[y1p:y2p, x1p:x2p]
                if crop.size > 0:
                    txt = run_ocr_on_crop(crop)
                    if txt:
                        texts.append(txt)

            # Merge unique lines from all OCR results
            if texts:
                merged = []
                seen = set()
                for block in texts:
                    for line in block.splitlines():
                        if line not in seen:
                            seen.add(line)
                            merged.append(line)
                extracted_text = "\n".join(merged)

            # ---------- Render ----------
            return render_template(
                'index.html',
                id_detected=id_detected,
                detection_confidence=detection_confidence,
                original_image=result_path.replace('\\', '/'),  # processed shown in both panes (as before)
                result_image=result_path.replace('\\', '/'),
                extracted_text=extracted_text
            )

        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('index.html')

@app.route('/submit_reason', methods=['POST'])
def submit_reason():
    try:
        reason = request.form.get('reason')
        if not reason:
            flash('Please select a reason', 'error')
            return redirect(url_for('index'))

        # Save to DB if needed
        flash(f'Reason submitted: {reason}. Thank you!', 'success')
        return redirect(url_for('index'))

    except Exception as e:
        flash(f'Error submitting reason: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # For VS Code debugger
    app.run(debug=True)
