from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.models import model_from_json
import numpy as np
import os
from PIL import Image
import base64
from io import BytesIO
import sqlite3
import pandas as pd
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max (increased)

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Model load karna
try:
    with open("emotiondetector.json", "r") as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights("emotiondetector.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    model = None

# Initialize database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id TEXT PRIMARY KEY,
                  filename TEXT,
                  emotion TEXT,
                  confidence REAL,
                  funny_text TEXT,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

init_db()

# Funny emotion labels
funny_labels = {
    'happy': 'Wah kia muskurahat hai! 😊',
    'neutral': 'Q nhi lag raha dil? 😐', 
    'sad': 'Q chor kr chali gayi kia? 😢'
}

# Emotion colors
emotion_colors = {
    'happy': '#FF6B6B',
    'neutral': '#4ECDC4',
    'sad': '#45B7D1'
}

# Emotion icons
emotion_icons = {
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢'
}

# Performance metrics
performance = {
    'training_accuracy': 85.50,
    'validation_accuracy': 82.30,
    'training_loss': 0.42,
    'validation_loss': 0.48,
    'epochs': 30,
    'dataset_size': '15,000+ images',
    'classes': 3
}

def save_to_db(filename, emotion, confidence, funny_text):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        pred_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        c.execute('''INSERT INTO predictions 
                     (id, filename, emotion, confidence, funny_text, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (pred_id, filename, emotion, confidence, funny_text, timestamp))
        
        conn.commit()
        conn.close()
        return pred_id
    except Exception as e:
        print(f"Database error: {e}")
        return None

def detect_multiple_faces(image_path):
    """Ek image mein multiple faces detect karta hai"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return []
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Face detection ka classifier load karo
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Faces detect karo
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        faces_data = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Har face ko crop karo
            face_img = img[y:y+h, x:x+w]
            
            # Face ko save karo temporary
            face_filename = f"face_{i}_{os.path.basename(image_path)}"
            face_path = os.path.join(app.config['UPLOAD_FOLDER'], face_filename)
            cv2.imwrite(face_path, face_img)
            
            faces_data.append({
                'index': i + 1,
                'x': x, 'y': y, 'w': w, 'h': h,
                'filename': face_filename,
                'path': face_path
            })
        
        return faces_data
    except Exception as e:
        print(f"Face detection error: {e}")
        return []

def preprocess_image(image_input):
    """Image ko model ke liye taiyar karna"""
    try:
        if isinstance(image_input, str):
            # Agar string hai toh check karo base64 ya file path
            if image_input.startswith('data:image'):
                # Base64 format
                image_data = image_input.split(',')[1]
                img = Image.open(BytesIO(base64.b64decode(image_data)))
            else:
                # File path
                img = Image.open(image_input)
        elif hasattr(image_input, 'read'):
            # File object
            img = Image.open(image_input)
        else:
            raise ValueError("Unsupported image input type")
        
        img = img.convert('L')  # Grayscale
        img = img.resize((48, 48))
        
        img_array = np.array(img)
        img_array = img_array.reshape(1, 48, 48, 1)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise e

@app.route('/')
def home():
    return render_template('index.html', 
                         performance=performance,
                         funny_labels=funny_labels,
                         emotion_icons=emotion_icons)

@app.route('/bulk-predict', methods=['POST'])
def bulk_predict():
    """Multiple images prediction"""
    try:
        if 'images[]' not in request.files:
            return jsonify({'success': False, 'error': 'No images provided'})
        
        files = request.files.getlist('images[]')
        all_results = []
        
        for file in files:
            if file.filename:
                # Save file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Multiple faces detect karo
                faces = detect_multiple_faces(filepath)
                face_results = []
                
                if faces:
                    # Har face ka prediction karo
                    for face in faces:
                        img_array = preprocess_image(face['path'])
                        predictions = model.predict(img_array, verbose=0)[0]
                        emotions = ['happy', 'neutral', 'sad']
                        
                        dominant_idx = np.argmax(predictions)
                        dominant_emotion = emotions[dominant_idx]
                        confidence = float(predictions[dominant_idx] * 100)
                        
                        # Save to database
                        pred_id = save_to_db(
                            face['filename'], 
                            dominant_emotion, 
                            confidence, 
                            funny_labels[dominant_emotion]
                        )
                        
                        face_results.append({
                            'id': pred_id,
                            'face_index': face['index'],
                            'filename': face['filename'],
                            'original_image': filename,
                            'emotion': dominant_emotion,
                            'funny_text': funny_labels[dominant_emotion],
                            'confidence': confidence,
                            'color': emotion_colors[dominant_emotion],
                            'icon': emotion_icons[dominant_emotion],
                            'position': {
                                'x': int(face['x']),
                                'y': int(face['y']),
                                'w': int(face['w']),
                                'h': int(face['h'])
                            }
                        })
                else:
                    # Agar face na mile, toh puri image analyze karo
                    img_array = preprocess_image(file)
                    predictions = model.predict(img_array, verbose=0)[0]
                    emotions = ['happy', 'neutral', 'sad']
                    
                    dominant_idx = np.argmax(predictions)
                    dominant_emotion = emotions[dominant_idx]
                    confidence = float(predictions[dominant_idx] * 100)
                    
                    pred_id = save_to_db(filename, dominant_emotion, 
                                       confidence, funny_labels[dominant_emotion])
                    
                    face_results.append({
                        'id': pred_id,
                        'face_index': 1,
                        'filename': filename,
                        'original_image': filename,
                        'emotion': dominant_emotion,
                        'funny_text': funny_labels[dominant_emotion],
                        'confidence': confidence,
                        'color': emotion_colors[dominant_emotion],
                        'icon': emotion_icons[dominant_emotion],
                        'position': None
                    })
                
                all_results.extend(face_results)
        
        return jsonify({
            'success': True,
            'total_faces': len(all_results),
            'total_images': len(files),
            'results': all_results
        })
        
    except Exception as e:
        print(f"Bulk predict error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict function called")  # Debug log
        
        # Check karo konsa format mein image aa rahi hai
        if 'image' in request.files:
            print("Image received as file")  # Debug log
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No selected file'})
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Multiple faces detect karo
            faces = detect_multiple_faces(filepath)
            results = []
            
            if faces:
                print(f"Found {len(faces)} faces")  # Debug log
                for face in faces:
                    try:
                        img_array = preprocess_image(face['path'])
                        predictions = model.predict(img_array, verbose=0)[0]
                        emotions = ['happy', 'neutral', 'sad']
                        
                        dominant_idx = np.argmax(predictions)
                        dominant_emotion = emotions[dominant_idx]
                        confidence = float(predictions[dominant_idx] * 100)
                        
                        pred_id = save_to_db(
                            face['filename'], 
                            dominant_emotion, 
                            confidence, 
                            funny_labels[dominant_emotion]
                        )
                        
                        results.append({
                            'id': pred_id,
                            'face_index': face['index'],
                            'filename': face['filename'],
                            'original_image': filename,
                            'emotion': dominant_emotion,
                            'funny_text': funny_labels[dominant_emotion],
                            'confidence': confidence,
                            'color': emotion_colors[dominant_emotion],
                            'icon': emotion_icons[dominant_emotion],
                            'position': {
                                'x': int(face['x']),
                                'y': int(face['y']),
                                'w': int(face['w']),
                                'h': int(face['h'])
                            }
                        })
                    except Exception as face_error:
                        print(f"Face processing error: {face_error}")
                        continue
            else:
                print("No faces found, processing whole image")  # Debug log
                # Agar face na mile
                img_array = preprocess_image(file)
                predictions = model.predict(img_array, verbose=0)[0]
                emotions = ['happy', 'neutral', 'sad']
                
                dominant_idx = np.argmax(predictions)
                dominant_emotion = emotions[dominant_idx]
                confidence = float(predictions[dominant_idx] * 100)
                
                pred_id = save_to_db(filename, dominant_emotion, confidence, funny_labels[dominant_emotion])
                
                results.append({
                    'id': pred_id,
                    'face_index': 1,
                    'filename': filename,
                    'original_image': filename,
                    'emotion': dominant_emotion,
                    'funny_text': funny_labels[dominant_emotion],
                    'confidence': confidence,
                    'color': emotion_colors[dominant_emotion],
                    'icon': emotion_icons[dominant_emotion],
                    'position': None
                })
            
            print(f"Returning {len(results)} results")  # Debug log
            
            # Dominant emotion find karo (highest confidence wala)
            dominant_result = None
            if results:
                dominant_result = max(results, key=lambda x: x['confidence'])
            
            return jsonify({
                'success': True,
                'total_faces': len(results),
                'all_faces': results,  # All faces results
                'main_result': dominant_result,  # Dominant emotion
                'has_multiple_faces': len(results) > 1
            })
            
        elif 'image_data' in request.form:
            print("Image received as base64 data")  # Debug log
            # Webcam image
            image_data = request.form['image_data']
            
            # Webcam ke liye sirf single face (real-time ke liye)
            try:
                img_array = preprocess_image(image_data)
                predictions = model.predict(img_array, verbose=0)[0]
                emotions = ['happy', 'neutral', 'sad']
                
                dominant_idx = np.argmax(predictions)
                dominant_emotion = emotions[dominant_idx]
                confidence = float(predictions[dominant_idx] * 100)
                
                # Save webcam image
                filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                img_data = image_data.split(',')[1]
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(img_data))
                
                pred_id = save_to_db(filename, dominant_emotion, confidence, funny_labels[dominant_emotion])
                
                result = {
                    'id': pred_id,
                    'face_index': 1,
                    'filename': filename,
                    'original_image': filename,
                    'emotion': dominant_emotion,
                    'funny_text': funny_labels[dominant_emotion],
                    'confidence': confidence,
                    'color': emotion_colors[dominant_emotion],
                    'icon': emotion_icons[dominant_emotion],
                    'position': None
                }
                
                return jsonify({
                    'success': True,
                    'total_faces': 1,
                    'all_faces': [result],
                    'main_result': result,
                    'has_multiple_faces': False
                })
            except Exception as e:
                print(f"Webcam prediction error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        else:
            return jsonify({'success': False, 'error': 'No image provided'})
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()  # Full error trace print karo
        return jsonify({'success': False, 'error': str(e)})

@app.route('/history')
def history():
    """View prediction history"""
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100')
    predictions = c.fetchall()
    conn.close()
    
    return render_template('history.html', predictions=predictions)

@app.route('/export-excel')
def export_excel():
    """Export data to Excel"""
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query('SELECT * FROM predictions ORDER BY timestamp DESC', conn)
    conn.close()
    
    excel_path = 'predictions_export.xlsx'
    df.to_excel(excel_path, index=False)
    
    return send_file(excel_path, as_attachment=True, download_name='emotion_predictions.xlsx')

@app.route('/export-csv')
def export_csv():
    """Export data to CSV"""
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query('SELECT * FROM predictions ORDER BY timestamp DESC', conn)
    conn.close()
    
    csv_path = 'predictions_export.csv'
    df.to_csv(csv_path, index=False)
    
    return send_file(csv_path, as_attachment=True, download_name='emotion_predictions.csv')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/bulk')
def bulk():
    return render_template('bulk.html')

@app.route('/live-webcam')
def live_webcam():
    return render_template('live_webcam.html')

if __name__ == '__main__':
    print("🚀 Starting Flask server at http://localhost:5000")
    app.run(debug=True, port=5000)