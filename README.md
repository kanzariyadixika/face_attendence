# Face Attendance Web App

A modern Face Attendance application using Flask, OpenCV (LBPH), and SQLite.

## Prerequisites

- Python 3.x installed.
- Pip package manager.

## Installation

1.  **Clone the repository** (or extract the zip):
    ```bash
    cd d:/face_attendence
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    ```bash
    python app.py
    ```

4.  **Open in Browser**:
    Go to `http://127.0.0.1:5000`

## Features

- **Registration**: Capture ~30 face images using webcam and train the model.
- **Attendance**: Real-time face recognition to mark attendance.
- **Dashboard**: View and manage attendance records.
- **Database**: SQLite `database.db`.

## Structure

- `app.py`: Main Flask application.
- `templates/`: HTML files.
- `static/`: CSS, JS, and Images.
- `haarcascade_frontalface_default.xml`: Pre-trained face detection model.

## Notes

- Ensure you have good lighting when registering.
- The model is retrained every time a new user registers.
