import os
import cv2
import numpy as np
from PIL import Image

path = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("Error loading xml file")
    exit(0)

# print([os.path.join(path, f) for f in os.listdir(path)])

imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
imagePath = imagePaths[0]

print(int(os.path.split(imagePath)[1]))

def getImageAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    print(imagePaths)
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        print(imagePath)
        for image in os.listdir(imagePath):
            image = os.path.join(imagePath, image)
            print(image)
            pil_img = Image.open(image).convert('L')
            img_numpy = np.array(pil_img, "uint8")
            # print(img_numpy)
            id = int(os.path.split(imagePath)[1])
            faces = face_cascade.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
    return faceSamples, ids

print("TRAINING")
faces, ids = getImageAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write("model/trainer.yml")
print("{0} faces trained. Existing Program".format(len(np.unique(ids))))