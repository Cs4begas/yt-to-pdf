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

def extract_frames(video_path, interval_seconds=15, ssim_threshold=0.70):
    """
    Extracts frames from a video at a given interval, avoiding duplicate frames.

    Args:
        video_path (str): The path to the video file.
        interval_seconds (int): The interval in seconds between each captured frame.
        ssim_threshold (float): The structural similarity threshold for duplicate detection.

    Returns:
        list: A list of paths to the extracted frames.
    """
    if not os.path.exists('frames'):
        os.makedirs('frames')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: Could not get FPS for video {video_path}. Assuming 30 FPS.")
        fps = 30

    interval_frames = int(fps * interval_seconds)

    frame_count = 0
    saved_frame_count = 0
    frame_paths = []
    last_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval_frames == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if last_frame is None or ssim(gray_frame, last_frame, data_range=gray_frame.max() - gray_frame.min()) < ssim_threshold:
                frame_path = f"frames/frame_{saved_frame_count}.jpg"
                height, width, _ = frame.shape
                resized_frame = cv2.resize(frame, (width // 2, height // 2))
                cv2.imwrite(frame_path, resized_frame)
                frame_paths.append(frame_path)
                saved_frame_count += 1
                last_frame = gray_frame

        frame_count += 1

    cap.release()
    return frame_paths

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
    frame_paths = extract_frames(video_path)
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
