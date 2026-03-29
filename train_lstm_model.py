import os
import glob
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    TimeDistributed,
    GlobalAveragePooling2D,
    LSTM,
    Dropout,
    Dense,
    Input,
    Bidirectional,
    BatchNormalization
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import confusion_matrix, classification_report

# ================= CONFIG =================

PROCESSED_DIR = r"D:\processed_data"

IMG_SIZE = (224, 224)
SEQ_LENGTH = 32
BATCH_SIZE = 4

EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 35

# ================= DATA GENERATOR =================

class VideoSequenceGenerator(tf.keras.utils.Sequence):

    def __init__(self, sequences, labels, batch_size, seq_length, img_size,
                 shuffle=True, augment=False):

        self.sequences = sequences
        self.labels = labels
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment

        self.indexes = np.arange(len(self.sequences))
        self.on_epoch_end()

    def __len__(self):
        return len(self.sequences) // self.batch_size

    def __getitem__(self, index):

        batch_ids = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X = np.zeros((self.batch_size, self.seq_length, *self.img_size, 3), dtype=np.float32)
        y = np.zeros((self.batch_size,), dtype=np.float32)

        for i, idx in enumerate(batch_ids):

            frames = self.sequences[idx]
            label = self.labels[idx]

            frame_indices = sorted(
                np.random.choice(len(frames), self.seq_length,
                replace=len(frames) < self.seq_length)
            )

            for j, frame_idx in enumerate(frame_indices):

                img = cv2.imread(frames[frame_idx])
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)

                if self.augment:

                    if np.random.rand() > 0.5:
                        img = cv2.flip(img, 1)

                    if np.random.rand() > 0.7:
                        angle = np.random.uniform(-10, 10)
                        M = cv2.getRotationMatrix2D((112,112), angle, 1)
                        img = cv2.warpAffine(img, M, (224,224))

                    if np.random.rand() > 0.7:
                        img = cv2.GaussianBlur(img, (3,3), 0)

                    if np.random.rand() > 0.6:
                        alpha = np.random.uniform(0.8,1.2)
                        img = np.clip(img * alpha,0,255)

                img = img.astype(np.float32)
                img = tf.keras.applications.xception.preprocess_input(img)

                X[i,j] = img

            y[i] = label

        return X,y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


# ================= DATA LOADING =================

def group_frames(folder):

    frames = glob.glob(os.path.join(folder,"*.jpg"))
    video_dict = {}

    for f in frames:

        vid = os.path.basename(f).split("_frame")[0]
        video_dict.setdefault(vid,[]).append(f)

    for vid in video_dict:
        video_dict[vid] = sorted(video_dict[vid])

    return list(video_dict.values())


def load_split(split):

    real = group_frames(os.path.join(PROCESSED_DIR,split,"real"))
    fake = group_frames(os.path.join(PROCESSED_DIR,split,"fake"))

    sequences = real + fake
    labels = [0]*len(real) + [1]*len(fake)

    combined = list(zip(sequences,labels))
    np.random.shuffle(combined)

    seqs,lbls = zip(*combined)

    return list(seqs), list(lbls)


train_seq,train_lbl = load_split("train")
val_seq,val_lbl = load_split("validation")
test_seq,test_lbl = load_split("test")

print("Train videos:",len(train_seq))
print("Validation videos:",len(val_seq))
print("Test videos:",len(test_seq))


train_gen = VideoSequenceGenerator(
    train_seq,train_lbl,BATCH_SIZE,SEQ_LENGTH,IMG_SIZE,
    shuffle=True,augment=True
)

val_gen = VideoSequenceGenerator(
    val_seq,val_lbl,BATCH_SIZE,SEQ_LENGTH,IMG_SIZE,
    shuffle=False,augment=False
)

test_gen = VideoSequenceGenerator(
    test_seq,test_lbl,BATCH_SIZE,SEQ_LENGTH,IMG_SIZE,
    shuffle=False,augment=False
)

# ================= MODEL =================

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

feature_extractor = Sequential([
    base_model,
    GlobalAveragePooling2D()
])

model = Sequential([

    Input(shape=(SEQ_LENGTH,224,224,3)),

    TimeDistributed(feature_extractor), 

    Bidirectional(
        LSTM(256,dropout=0.4,recurrent_dropout=0.4)
    ),

    BatchNormalization(),

    Dense(128,activation="relu"),

    Dropout(0.5),

    Dense(1,activation="sigmoid")

])

model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy",tf.keras.metrics.AUC(name="auc")]
)

model.summary()

# ================= PHASE 1 TRAINING =================

callbacks1 = [

    EarlyStopping(patience=6,restore_best_weights=True),

    ReduceLROnPlateau(patience=3),

    ModelCheckpoint("xception_phase1.keras",save_best_only=True)

]

print("Starting PHASE 1")

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks1
)

# ================= PHASE 2 FINETUNE =================

base_model.trainable = True

for layer in base_model.layers[:-80]:
    layer.trainable = False

model.compile(
    optimizer=Adam(3e-6),
    loss="binary_crossentropy",
    metrics=["accuracy",tf.keras.metrics.AUC(name="auc")]
)

callbacks2 = [

    EarlyStopping(patience=5,restore_best_weights=True),

    ReduceLROnPlateau(patience=3),

    ModelCheckpoint("xception_final.keras",save_best_only=True)

]

print("Starting PHASE 2")

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks2
)

# ================= SAVE MODEL =================

model.save("deepfake_xception_lstm.keras")

print("Final model saved")
