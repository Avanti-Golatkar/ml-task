import os
import cv2
import numpy as np
from tqdm import tqdm
print("PREPROCESS SCRIPT STARTED")

# Config
DATA_DIR = "data/train"
IMG_SIZE = 224
FRAMES_PER_VIDEO = 5

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while len(frames) < FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 10 == 0:  # take every 10th frame
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)

        count += 1

    cap.release()
    return frames


def load_data():
    X = []
    y = []

    for label, category in enumerate(["real", "fake"]):
        folder_path = os.path.join(DATA_DIR, category)

        for video_file in tqdm(os.listdir(folder_path), desc=f"Processing {category}"):
            video_path = os.path.join(folder_path, video_file)
            frames = extract_frames(video_path)

            for frame in frames:
                X.append(frame)
                y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_data()
    print("Data loaded:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.save("X.npy", X)
    np.save("y.npy", y)
