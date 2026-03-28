import os
import cv2
import glob
import logging
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN

SOURCE_DIR = r"D:\data"
PROCESSED_DIR = r"D:\processed_new"

SPLITS = ["train", "test", "validation"]
CATEGORIES = ["real", "fake"]


FRAMES_PER_VIDEO = 20
IMG_SIZE = (224, 224)

detector = MTCNN()

logging.basicConfig(
    filename='preprocessing_errors.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def setup_dirs():
    for split in SPLITS:
        for category in CATEGORIES:
            os.makedirs(
                os.path.join(PROCESSED_DIR, split, category),
                exist_ok=True
            )

def extract_faces(video_path, output_dir):
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            return False

        # Oversample then stop at 20 faces
        frame_indices = np.linspace(
            0,
            total_frames - 1,
            min(total_frames, FRAMES_PER_VIDEO * 4),
            dtype=int
        )

        video_name = os.path.basename(video_path).split('.')[0]
        faces_saved = 0

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb)

            if len(detections) > 0:
                detections = sorted(
                    detections,
                    key=lambda x: x['box'][2] * x['box'][3],
                    reverse=True
                )

                x, y, w, h = detections[0]['box']
                x, y, w, h = abs(int(x)), abs(int(y)), int(w), int(h)

                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)

                H, W = frame.shape[:2]
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(W, x + w + pad_x)
                y2 = min(H, y + h + pad_y)

                face = frame[y1:y2, x1:x2]

                if face.shape[0] > 0 and face.shape[1] > 0:
                    face = cv2.resize(face, IMG_SIZE)
                    save_path = os.path.join(
                        output_dir,
                        f"{video_name}_frame{faces_saved:02d}.jpg"
                    )
                    cv2.imwrite(save_path, face)
                    faces_saved += 1

                if faces_saved >= FRAMES_PER_VIDEO:
                    break

        cap.release()
        return True

    except Exception as e:
        logging.error(f"{video_path} - {e}")
        return False

def main():
    setup_dirs()

    for split in SPLITS:
        for category in CATEGORIES:

            video_folder = os.path.join(SOURCE_DIR, split, category)
            output_folder = os.path.join(PROCESSED_DIR, split, category)

            videos = []
            for ext in ('*.mp4', '*.avi', '*.mov'):
                videos.extend(glob.glob(os.path.join(video_folder, ext)))

            if not videos:
                continue

            for video in tqdm(videos, desc=f"{split}/{category}"):
                extract_faces(video, output_folder)

    print(" Preprocessing Complete")

if __name__ == "__main__":
    main()