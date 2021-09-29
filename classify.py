from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os


# 명령줄 인수를 파싱해옵니다
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# 이미지를 로드합니다
image = cv2.imread(args["image"])
output = imutils.resize(image, width=400)

# 분류를 위한 이미지 전처리를 수행합니다
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# 학습된 네트워크와 `MultiLabelBinarizer`를 로드합니다
print("[INFO] loading network...")
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())

# 이미지에 대한 분류를 수행한 후, 
# 확률이 가장 높은 두 개의 클래스 라벨을 찾습니다
print("[INFO] classifying image...")
proba = model.predict(image)[0]

# 각 라벨에 대한 확률을 출력합니다
finalImgLabelList=[]
for (label, p) in zip(mlb.classes_, proba):
	# if p > 0.1:
	# 	print("{}: {:.2f}%".format(label, p * 100))
	if p == max(proba):
		print(label)