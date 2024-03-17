import cv2
import os
import numpy as np
import time

# Inisialisasi detektor wajah deep learning
prototxtPath = r"CAFFE_DNN/deploy.prototxt.txt"
weightsPath = r"CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel"

face_net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# Inisialisasi webcam (ganti 0 dengan lokasi video jika menggunakan file video)
video_capture = cv2.VideoCapture(0)

# Input user ID (face_id dan nama)
face_id = input('\n Enter user ID (face_id) and press <return> ==>  ')
name = input('\n Enter user name and press <return> ==>  ')

# Membaca isi file labels.txt dan memeriksa apakah ada duplikat
existing_labels = set()
with open("dataset/labels.txt", "r") as label_file:
    for line in label_file:
        line = line.strip()
        if line:
            face_id_existing, _ = line.split(',')
            existing_labels.add(face_id_existing)

# Memeriksa apakah face_id sudah ada dalam file labels.txt
if face_id in existing_labels:
    print(f"User ID '{face_id}' already exists. Please choose a different ID.")
else:
    # Menambahkan data baru ke dalam file labels.txt
    with open("labels.txt", "a") as label_file:
        label_file.write(f"{face_id},{name}\n")
        print(f"User ID '{face_id}' added successfully.")

# Buat folder sesuai dengan user ID (face_id) jika belum ada
output_folder = os.path.join("dataset", face_id)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

# Loop untuk dua tahap (tanpa masker dan dengan masker)
for stage in range(2):
    if stage == 0:
        print("\n [INFO] Starting capture without a mask. Face straight, left, and right.")
    else:
        print("\n [INFO] Starting capture with a mask. Face straight, left, and right.")

    for orientation in range(3):  # 0: straight, 1: left, 2: right
        if orientation == 0:
            print("\n [INFO] Look straight.")
        elif orientation == 1:
            print("\n [INFO] Turn left.")
        else:
            print("\n [INFO] Turn right.")

        count = 0
        while True:
            # Baca frame dari video
            ret, frame = video_capture.read()
            
            # Konversi ke grayscale untuk deteksi wajah
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = frame

            # Mendapatkan tinggi dan lebar frame
            (h, w) = frame.shape[:2]

            # Pra-pemrosesan frame
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123))

            # Melewati frame ke model deteksi wajah
            face_net.setInput(blob)
            detections = face_net.forward()

            # Gambar kotak di sekitar wajah yang terdeteksi
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Filter deteksi wajah dengan tingkat kepercayaan yang cukup
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Gambar kotak di sekitar wajah
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    
                    if stage == 0:
                        label = "no_mask"
                    else:
                        label = "mask"

                    if orientation == 0:
                        label += "_straight"
                    elif orientation == 1:
                        label += "_left"
                    else:
                        label += "_right"

                    count += 1
                    img_name = os.path.join(output_folder, f"{label}_{count}.jpg")
                    cv2.imwrite(img_name, gray[startY:endY, startX:endX])
                    # cv2.imshow('img', gray[startY:endY, startX:endX])

            # Tampilkan frame dengan kotak wajah
            cv2.imshow('Video', frame)

            # Tekan 'q' untuk keluar dari loop
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif count >= 60:  # Ambil 10 sampel wajah dan hentikan video
                break

    print("\n ===================jeda sebelum pindah pose==========\n")
    time.sleep(5)

    if stage == 0:
        print("\n [INFO] Capture without a mask completed.")
    else:
        print("\n [INFO] Capture with a mask completed.")

# Tunggu sebentar sebelum menutup video capture
time.sleep(1)

# Tutup video capture dan jendela OpenCV
try:
    # Release video capture
    if video_capture.isOpened():
        video_capture.release()
except Exception as e:
    print("Error releasing video capture:", e)

try:
    # Close OpenCV windows
    cv2.destroyAllWindows()
except Exception as e:
    print("Error closing OpenCV windows:", e)
