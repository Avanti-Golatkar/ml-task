import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


MODEL_PATH = "fake_real_video_model.keras"
TEST_CSV = "data/test_public.csv"
VIDEO_DIR = "data/test"
IMG_SIZE = 224
FRAMES_PER_VIDEO = 10
OUTPUT_CSV = "predictions.csv"



def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    frame_idxs = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)

    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    return np.mean(frames, axis=0)


def main():
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Loading test CSV...")
    test_df = pd.read_csv(TEST_CSV)

    results = []

    for filename in tqdm(test_df["filename"]):
        video_path = os.path.join(VIDEO_DIR, filename)

        if not os.path.exists(video_path):
            results.append([filename, 0, 0.0])
            continue

        frame = extract_frames(video_path)

        if frame is None:
            results.append([filename, 0, 0.0])
            continue

        frame = np.expand_dims(frame, axis=0)
        prob = model.predict(frame, verbose=0)[0][0]
        prediction = 1 if prob >= 0.5 else 0

        results.append([filename, prediction, float(prob)])

    output_df = pd.DataFrame(
        results,
        columns=["Video_Name", "Prediction", "Probability"]
    )

    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
