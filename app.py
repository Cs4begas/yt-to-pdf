import cv2
from fpdf import FPDF
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import re

def natural_sort_key(s):
    """
    Key for natural sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def calculate_ssim(frame1, frame2):
    """Calculates the structural similarity between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def calculate_content_density(frame):
    """Calculates the content density of a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.count_nonzero(edges)
    total_pixels = frame.shape[0] * frame.shape[1]
    density = edge_count / total_pixels
    return density

def calculate_text_area(frame):
    """Estimates the area of text in a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    text_pixels = np.count_nonzero(binary)
    total_pixels = frame.shape[0] * frame.shape[1]
    text_ratio = text_pixels / total_pixels
    return text_ratio

def multi_stage_capture(video_path, frame_interval=3):
    """3-stage capture: Collect, Filter, Select."""

    # Stage 1: Collect frames every 3 seconds
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: Could not get FPS for video {video_path}. Assuming 30 FPS.")
        fps = 30

    frame_step = int(fps * frame_interval)

    raw_frames = []
    frame_count = 0

    print("Stage 1: Collecting frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_step == 0:
            timestamp = frame_count / fps
            raw_frames.append({
                'frame': frame.copy(),
                'timestamp': timestamp,
                'content_score': calculate_content_density(frame) + calculate_text_area(frame)
            })

        frame_count += 1

    cap.release()
    print(f"Collected {len(raw_frames)} raw frames")

    # Stage 2: Filter distinct frames (SSIM < 0.60)
    print("Stage 2: Filtering distinct frames...")
    distinct_frames = []

    for i, candidate in enumerate(raw_frames):
        is_distinct = True

        for saved in distinct_frames:
            ssim_score = calculate_ssim(candidate['frame'], saved['frame'])
            if ssim_score > 0.60:
                is_distinct = False
                break

        if is_distinct:
            distinct_frames.append(candidate)

    print(f"Filtered to {len(distinct_frames)} distinct frames")

    # Stage 3: Select best quality frames from similar groups
    print("Stage 3: Selecting best quality frames...")
    final_frames = []

    for candidate in distinct_frames:
        time_window = 10
        nearby_frames = [
            f for f in raw_frames
            if abs(f['timestamp'] - candidate['timestamp']) <= time_window
        ]

        if nearby_frames:
            best_frame = max(nearby_frames, key=lambda x: x['content_score'])
            # Avoid adding duplicate frames
            if not any(np.array_equal(best_frame['frame'], f['frame']) for f in final_frames):
                final_frames.append(best_frame)

    print(f"Final selection: {len(final_frames)} frames")

    return [f['frame'] for f in final_frames], [f['timestamp'] for f in final_frames]


def create_pdf_from_images(image_paths, pdf_path):
    """
    Creates a PDF file from a list of images, with two images per page.
    """
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    for i in range(0, len(image_paths), 2):
        pdf.add_page()
        if os.path.exists(image_paths[i]):
            pdf.image(image_paths[i], x=10, y=10, w=190)
        if i + 1 < len(image_paths) and os.path.exists(image_paths[i+1]):
            pdf.image(image_paths[i+1], x=10, y=150, w=190)

    pdf.output(pdf_path, "F")

def process_video(video_path):
    """
    Processes a single video file.
    """
    if not os.path.exists('frames'):
        os.makedirs('frames')

    saved_frames, _ = multi_stage_capture(video_path)

    frame_paths = []
    for i, frame in enumerate(saved_frames):
        frame_path = f"frames/frame_{i}.jpg"
        height, width, _ = frame.shape
        resized_frame = cv2.resize(frame, (width // 2, height // 2))
        cv2.imwrite(frame_path, resized_frame)
        frame_paths.append(frame_path)

    if frame_paths:
        pdf_path = f"{os.path.splitext(os.path.basename(video_path))[0]}.pdf"
        create_pdf_from_images(frame_paths, pdf_path)
        print(f"Successfully created PDF for {os.path.basename(video_path)} at {pdf_path}")

def process_videos_from_folder(folder_path):
    """
    Processes all video files in a given folder, sorted naturally.
    """
    print(f"Processing videos from folder: {folder_path}")
    filenames = sorted(os.listdir(folder_path), key=natural_sort_key)
    print(f"Found {len(filenames)} files.")
    for filename in filenames:
        if filename.endswith((".mp4", ".avi", ".mov", ".webm", ".mkv")):
            video_path = os.path.join(folder_path, filename)
            print(f"Processing video: {video_path}")
            process_video(video_path)

    messagebox.showinfo("Success", "All videos have been processed.")

def select_file():
    """
    Opens a file dialog to select a single video file.
    """
    filepath = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=(("Video Files", "*.mp4 *.avi *.mov *.webm *.mkv"), ("All files", "*.*"))
    )
    if filepath:
        process_video(filepath)

def select_folder():
    """
    Opens a dialog to select a folder containing video files.
    """
    folderpath = filedialog.askdirectory(title="Select a Folder with Videos")
    print(f"Selected folder: {folderpath}")
    if folderpath:
        process_videos_from_folder(folderpath)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Video to PDF Converter")
    root.geometry("300x150")

    btn_select_file = tk.Button(root, text="Select Video File", command=select_file)
    btn_select_file.pack(pady=10)

    btn_select_folder = tk.Button(root, text="Select Folder", command=select_folder)
    btn_select_folder.pack(pady=10)

    root.mainloop()
