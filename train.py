import cv2
import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Input, Activation, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator

# Path untuk dataset gambar wajah (setiap folder berisi gambar wajah satu orang)
dataset_path = 'dataset'

# Inisialisasi lists untuk gambar wajah dan label-label
faceSamples = []
labels = []

# Loop melalui folder di dalam dataset_path
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    # Dapatkan label dari nama folder (nama orang)
    label = folder_name
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Resize gambar ke ukuran yang sama
        img = img / 255.0  # Normalize pixel values to [0, 1]
        faceSamples.append(img)
        labels.append(label)

# Konversi lists ke numpy arrays
faceSamples = np.array(faceSamples)
labels = np.array(labels)

# Encode label-label menjadi nilai numerik
le = LabelEncoder()
labels = le.fit_transform(labels)

# One-hot encoding label-label
labels = np_utils.to_categorical(labels)

# Split dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(faceSamples, labels, test_size=0.2, random_state=42)

# Data Augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Inisialisasi model CNN
# Load pre-trained VGGFace model
faceNet = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Freeze semua lapisan VGGFace agar tidak terlatih kembali
for layer in faceNet.layers:
    layer.trainable = False

# Inisialisasi model baru
# model = Sequential()

# # Tambahkan VGGFace sebagai lapisan pertama (tanpa top layer)
# model.add(faceNet)

# # Tambahkan lapisan-lapisan berikutnya sesuai dengan kebutuhan Anda
# model.add(GlobalAveragePooling2D())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(le.classes_), activation='softmax'))

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_)))
model.add(Activation('softmax'))


# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

# Latih model dengan data augmentation
model.fit(datagen.flow(X_train, y_train, batch_size=16), epochs=10, validation_data=(X_test, y_test))

# Simpan model
model.save('face_recognition_model.h5')
