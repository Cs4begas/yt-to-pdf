import cv2
from fpdf import FPDF
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np


def extract_frames(video_path, interval_minutes=1):
    """
    Extracts frames from a video at a given interval, avoiding duplicate frames.

    Args:
        video_path (str): The path to the video file.
        interval_minutes (int): The interval in minutes between each captured frame.

    Returns:
        list: A list of paths to the extracted frames.
    """
    if not os.path.exists('frames'):
        os.makedirs('frames')

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_minutes * 60)

    frame_count = 0
    saved_frame_count = 0
    frame_paths = []
    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval_frames == 0:
            # Convert frame to grayscale for SSIM
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if last_frame is None or ssim(gray_frame, last_frame, data_range=gray_frame.max() - gray_frame.min()) < 0.95:
                frame_path = f"frames/frame_{saved_frame_count}.jpg"
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_frame_count += 1
                last_frame = gray_frame

        frame_count += 1

    cap.release()
    return frame_paths

def create_pdf_from_images(image_paths, pdf_path):
    """
    Creates a PDF file from a list of images.

    Args:
        image_paths (list): A list of paths to the image files.
        pdf_path (str): The path to save the output PDF file.
    """
    pdf = FPDF()
    for image_path in image_paths:
        pdf.add_page()
        pdf.image(image_path, x=10, y=8, w=190)

    pdf.output(pdf_path, "F")

from flask import Flask, request, send_file, render_template_string
import werkzeug.utils

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = werkzeug.utils.secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            frame_paths = extract_frames(video_path, interval_minutes=1)
            pdf_path = "output.pdf"
            create_pdf_from_images(frame_paths, pdf_path)

            return send_file(pdf_path, as_attachment=True)

    return '''
    <!doctype html>
    <title>Upload a Video to Convert to PDF</title>
    <h1>Upload a Video to Convert to PDF</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
