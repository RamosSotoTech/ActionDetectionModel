import cv2
import tkinter as tk
from PIL import Image, ImageTk
from dataset_generators import get_dataset_generators
import numpy as np


class GUI:
    def __init__(self, iterator):
        self.iterator = iterator
        self.root = tk.Tk()
        self.root.geometry('500x500')  # Sets root window size.

        self.label = tk.Label(self.root)
        self.label.pack()

        # Bind the spacebar key and the ctrl key with the video_loop function
        self.root.bind('<space>', self.video_loop)  # Moves one frame forward
        self.root.bind('<Control-Key>', self.next_video)  # Continues until next video

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handles window closing.
        # Start the video process
        self.root.after(0, self.video_loop)
        self.root.mainloop()

    def video_loop(self, event=None):
        # Function to display the video frames
        try:
            frame = next(self.iterator)

            # Check the type and shape of the frame
            if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR format, PIL uses RGB
                frame = Image.fromarray(frame)  # Convert the image to a PIL object
                frame = ImageTk.PhotoImage(frame)  # Convert the image to a tkinter object
                self.label.configure(image=frame)  # Set the image to the label
                self.label.image = frame  # Keep a reference
                self.root.after(0, self.video_loop)  # Repeat the function
            else:
                print("Frame is not an image.")
        except StopIteration:
            print("End of video")
            self.root.destroy()

    def next_video(self, event=None):
        # Implement function to continue until next video.
        try:
            self.iterator = iter(self.iterator)
            next(self.iterator)
        except StopIteration:
            print("No more videos")

    def on_closing(self):
        # Function to handle window closing.
        self.root.destroy()  # Destroys the tkinter root window.


def data_iterator():
    # Create a generator for the UCF-101 dataset
    train_generator = get_dataset_generators([1], batch_size=1, preprocessing=None, sequence_length=1)[0]
    return train_generator


if __name__ == "__main__":
    GUI(data_iterator())