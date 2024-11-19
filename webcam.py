import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import logging as log
import datetime as dt
from PIL import Image, ImageTk
import ollama
import os
import customtkinter as ctk
from pathlib import Path
import threading
import time
import numpy as np


class ModernFaceComplimentApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Face Compliment AI")
        self.root.geometry("1400x900")
        self.root.configure(fg_color="#1a1a1a")

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        log.basicConfig(filename='app.log', level=log.INFO)
        self.cascPath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.video_capture = None
        self.is_camera_active = False
        self.is_processing = False

        self.setup_ui()

    def setup_ui(self):
        # Main container
        self.container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=20, pady=20)

        # Left Panel
        self.left_panel = ctk.CTkFrame(self.container, fg_color="#2d2d2d", corner_radius=15)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # Camera frame
        self.camera_frame = ctk.CTkFrame(self.left_panel, fg_color="#1a1a1a", corner_radius=10)
        self.camera_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.video_label = tk.Label(self.camera_frame, bg="#1a1a1a")
        self.video_label.pack(pady=10)

        # Progress bar
        self.progress = ctk.CTkProgressBar(self.left_panel, mode="indeterminate")
        self.progress.pack(padx=20, pady=(0, 20), fill="x")
        self.progress.set(0)

        # Right Panel
        self.right_panel = ctk.CTkFrame(self.container, width=300, fg_color="#2d2d2d", corner_radius=15)
        self.right_panel.pack(side="right", fill="y", padx=(10, 0))

        # Title
        title_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        title_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(title_frame, text="Face Compliment AI",
                     font=ctk.CTkFont(size=24, weight="bold")).pack()

        # Buttons
        self.buttons_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.buttons_frame.pack(fill="x", padx=20, pady=10)

        self.camera_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Start Camera",
            command=self.toggle_camera,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#4CAF50",
            hover_color="#388E3C"
        )
        self.camera_btn.pack(fill="x", pady=5)

        self.upload_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Upload Image",
            command=self.upload_image,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2196F3",
            hover_color="#1976D2"
        )
        self.upload_btn.pack(fill="x", pady=5)

        self.capture_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Capture & Analyze",
            command=self.capture_and_analyze,
            height=40,
            corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#9C27B0",
            hover_color="#7B1FA2"
        )
        self.capture_btn.pack(fill="x", pady=5)

        # Status frame
        self.status_frame = ctk.CTkFrame(self.right_panel, fg_color="#232323", corner_radius=10)
        self.status_frame.pack(fill="x", padx=20, pady=10)

        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready to start...",
            font=ctk.CTkFont(size=12),
            wraplength=250
        )
        self.status_label.pack(padx=15, pady=15)

        # Result frame
        self.result_frame = ctk.CTkFrame(self.right_panel, fg_color="#232323", corner_radius=10)
        self.result_frame.pack(fill="both", expand=True, padx=20, pady=(10, 20))

        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Compliments will appear here...",
            font=ctk.CTkFont(size=14),
            wraplength=250
        )
        self.result_label.pack(padx=15, pady=15, fill="both", expand=True)

    def update_status(self, message):
        self.root.after(0, lambda: self.status_label.configure(text=message))

    def update_result(self, message):
        self.root.after(0, lambda: self.result_label.configure(text=message))

    def animate_processing(self):
        self.progress.start()
        self.update_status("Processing...")

    def stop_animation(self):
        self.progress.stop()
        self.update_status("Ready")

    def toggle_camera(self):
        if not self.is_camera_active:
            self.video_capture = cv2.VideoCapture(0)
            if self.video_capture.isOpened():
                self.is_camera_active = True
                self.camera_btn.configure(
                    text="Stop Camera",
                    fg_color="#f44336",
                    hover_color="#d32f2f"
                )
                self.update_status("Camera active")
                self.update_video()
        else:
            if self.video_capture is not None:
                self.video_capture.release()
            self.is_camera_active = False
            self.camera_btn.configure(
                text="Start Camera",
                fg_color="#4CAF50",
                hover_color="#388E3C"
            )
            self.update_status("Camera stopped")
            self.video_label.configure(image='')

    def update_video(self):
        if self.is_camera_active:
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame

                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # Draw rectangles
                for (x, y, w, h) in faces:
                    thickness = 2 + int(abs(np.sin(time.time() * 4)) * 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness)

                frame = cv2.addWeighted(frame, 0.9, frame, 0.1, 0)

                # Convert to tkinter image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image = image.resize((800, 600), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=image)

                self.video_label.configure(image=photo)
                self.video_label.image = photo

            self.root.after(10, self.update_video)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            self.current_frame = cv2.imread(file_path)
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = image.resize((800, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=image)

            self.video_label.configure(image=photo)
            self.video_label.image = photo
            self.update_status("Image uploaded")

    def capture_and_analyze(self):
        if self.current_frame is not None and not self.is_processing:
            self.is_processing = True
            threading.Thread(target=self._process_image).start()
            self.animate_processing()

    def _process_image(self):
        try:
            filename = f"captured_image_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, self.current_frame)

            response = self.describe_image(filename)
            self.update_result(response)
            self.update_status("Analysis complete!")

            if os.path.exists(filename):
                os.remove(filename)

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.update_result("An error occurred during analysis")
        finally:
            self.is_processing = False
            self.stop_animation()

    def describe_image(self, image_path, max_line_length=35):
        res = ollama.chat(
            model="llava:7b",
            messages=[{
                'role': 'user',
                'content': 'Compliment features of my face with a short response:',
                'images': [image_path]
            }]
        )
        content = res['message']['content']
        lines = []
        current_line = ''
        for word in content.split():
            if len(current_line) + len(word) + 1 <= max_line_length:
                current_line += f"{word} "
            else:
                lines.append(current_line.strip())
                current_line = f"{word} "
        if current_line:
            lines.append(current_line.strip())
        return '\n'.join(lines)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = ModernFaceComplimentApp()
    app.run()