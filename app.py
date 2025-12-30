import cv2
import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import sqlite3

app = Flask(__name__)

# --- Configuration ---
DATADIR = "static/images"
MODEL_FILE = "trainer.yml"
DB_FILE = "database.db"

if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

# --- Database Setup ---
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)''')
    # Attendance table
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  user_id INTEGER, 
                  date TEXT, 
                  time TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# --- Face Recognition Models ---
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Initialize LBPH Face Recognizer
# Check if cv2.face is available (requires opencv-contrib-python)
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("ERROR: cv2.face not available. Install opencv-contrib-python")
    exit(1)

# Load trained model if exists
if os.path.exists(MODEL_FILE):
    try:
        recognizer.read(MODEL_FILE)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")

# --- Helper Functions ---
def train_model():
    faces = []
    ids = []
    
    # Iterate through all user folders in static/images
    if not os.path.exists(DATADIR):
        return
        
    user_folders = [f for f in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, f))]
    
    for user_id in user_folders:
        path = os.path.join(DATADIR, user_id)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # We assume images stored are already cropped faces, but good to check
                # For training, we need consistent sizes, or LBPH handles it? LBPH handles it mostly but resizing is good.
                # Here we just pass the image.
                faces.append(img)
                ids.append(int(user_id))
            except Exception as e:
                print(f"Error reading {img_path}: {e}")

    if faces and ids:
        print(f"Training on {len(faces)} images for {len(set(ids))} users...")
        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_FILE)
        print("Model trained and saved.")
    else:
        print("No data to train.")

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    
    # Get attendance with user names
    # Using LEFT JOIN to show attendance even if user deleted (though we cascade delete usually)
    query = '''SELECT a.id, u.name, a.date, a.time, a.user_id 
               FROM attendance a 
               JOIN users u ON a.user_id = u.id 
               ORDER BY a.date DESC, a.time DESC'''
    attendance = conn.execute(query).fetchall()
    conn.close()
    
    return render_template('dashboard.html', users=users, attendance=attendance)

# API to create user entry in DB
@app.route('/api/create_user', methods=['POST'])
def create_user():
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('INSERT INTO users (name) VALUES (?)', (name,))
    user_id = c.lastrowid
    conn.commit()
    conn.close()
    
    # Create folder for user
    os.makedirs(os.path.join(DATADIR, str(user_id)), exist_ok=True)
    
    return jsonify({'user_id': user_id, 'message': 'User created'})

# API to upload face images for training
@app.route('/api/upload_face', methods=['POST'])
def upload_face():
    user_id = request.form.get('user_id')
    image_file = request.files.get('image')
    
    if not user_id or not image_file:
        return jsonify({'error': 'Missing data'}), 400

    # Save image
    # We can use a timestamp or counter for filename
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    save_path = os.path.join(DATADIR, str(user_id), filename)
    
    # We need to detect face and save ONLY the face part for better training (usually)
    # Or front-end sends full image and we crop here.
    # Let's save the received file first, then try to detect/crop.
    image_file.save(save_path)
    
    # Post-processing: Crop face
    img = cv2.imread(save_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        (x,y,w,h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(save_path, face_img) # Overwrite with cropped face
        return jsonify({'status': 'success', 'message': 'Face detected and saved'})
    else:
        # If no face detected, maybe delete the file?
        os.remove(save_path)
        return jsonify({'status': 'retry', 'message': 'No face detected'}), 400

# API to trigger training
@app.route('/api/train', methods=['POST'])
def trigger_train():
    try:
        train_model()
        return jsonify({'status': 'success', 'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API for Marking Attendance
@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image'}), 400
        
    # Save temp file
    temp_path = "temp_attendance.jpg"
    image_file.save(temp_path)
    
    img = cv2.imread(temp_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    os.remove(temp_path) # Clean up
    
    if len(faces) == 0:
         return jsonify({'status': 'fail', 'message': 'No face detected'})
         
    # Take the largest face
    # (x,y,w,h) = max(faces, key=lambda b: b[2]*b[3]) # Width * Height
    # For now just take the first
    (x,y,w,h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    if not os.path.exists(MODEL_FILE):
         return jsonify({'status': 'fail', 'message': 'Model not trained yet'})

    label, confidence = recognizer.predict(face_roi)
    
    # Confidence: 0 is perfect match. Usually < 50 is good, < 100 is acceptable.
    # Adjust threshold as needed.
    if confidence < 70:
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE id = ?', (label,)).fetchone()
        
        if user:
            user_name = user['name']
            
            # Check for duplicate attendance today
            today = date.today().strftime("%Y-%m-d")
            existing = conn.execute('SELECT * FROM attendance WHERE user_id = ? AND date = ?', 
                                    (label, today)).fetchone()
            
            if existing: # Commenting this out for easy testing if needed, or keeping it strict
                 conn.close()
                 return jsonify({'status': 'success', 'match': True, 'user': user_name, 'message': 'Already marked today'})
            
            now_time = datetime.now().strftime("%H:%M:%S")
            conn.execute('INSERT INTO attendance (user_id, date, time) VALUES (?, ?, ?)',
                         (label, today, now_time))
            conn.commit()
            conn.close()
            
            return jsonify({'status': 'success', 'match': True, 'user': user_name, 'message': 'Attendance marked'})
    
    return jsonify({'status': 'success', 'match': False, 'message': 'Face not recognized'})

@app.route('/delete_user/<int:id>')
def delete_user(id):
    conn = get_db_connection()
    conn.execute('DELETE FROM users WHERE id = ?', (id,))
    conn.execute('DELETE FROM attendance WHERE user_id = ?', (id,))
    conn.commit()
    conn.close()
    
    # Remove folder
    # Implementation optional: import shutil; shutil.rmtree...
    import shutil
    user_dir = os.path.join(DATADIR, str(id))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
        
    # Retrain model
    train_model()
    
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
