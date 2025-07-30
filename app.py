import cv2
from fpdf import FPDF
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

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
            if last_frame is None or ssim(gray_frame, last_frame, data_range=gray_frame.max() - gray_frame.min()) < 0.90:
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
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    for image_path in image_paths:
        pdf.add_page()
        pdf.image(image_path, x=0, y=0, w=297, h=210)

    pdf.output(pdf_path, "F")

def process_video(video_path):
    """
    Processes a single video file.
    """
    frame_paths = extract_frames(video_path, interval_minutes=1)
    if frame_paths:
        pdf_path = f"{os.path.splitext(os.path.basename(video_path))[0]}.pdf"
        create_pdf_from_images(frame_paths, pdf_path)
        messagebox.showinfo("Success", f"Created PDF for {os.path.basename(video_path)} at {pdf_path}")

def process_videos_from_folder(folder_path):
    """
    Processes all video files in a given folder, sorted by name.
    """
    print(f"Processing videos from folder: {folder_path}")
    filenames = sorted(os.listdir(folder_path))
    print(f"Found {len(filenames)} files.")
    for filename in filenames:
        if filename.endswith((".mp4", ".avi", ".mov", ".webm", ".mkv")):
            video_path = os.path.join(folder_path, filename)
            print(f"Processing video: {video_path}")
            process_video(video_path)

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
