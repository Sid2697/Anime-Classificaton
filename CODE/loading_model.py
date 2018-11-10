# importing necessary packages
from keras.models import load_model
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.preprocessing import AspectAwarePreprocessor, ImageToArrayPreprocessor, SimplePreprocessor
from preprocessing.datasets import SimpleDatasetLoader
from keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import glob
import cv2
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', help='path to the saved model', default='/home/stu15/s15/ts6442/Capstone/codes/third_model_final.h5')
args = vars(ap.parse_args())

model = load_model(args['path'])
model.summary()

# grab the list of images
print('[INFO] loading images...')
imagePaths = glob.glob('/home/stu15/s15/ts6442/Capstone/images/images/*.jpg')

# Resize the image keeping aspect ratio in mind
aap = AspectAwarePreprocessor(128, 128)
# Resize the image without aspect ratio in mind
# sp = SimplePreprocessor(128, 128)
# converting images to array for easier processing
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
# as there are no labels using '_' in place of labels
(data, _) = sdl.load(imagePaths, verbose=1000)
data = data.astype('float') / 255.0
print('[INFO] total number of images are ', len(data))
print('Shape of data is', data.shape)

(trainX, testX, _, _) = train_test_split(data, _, test_size=0.05)
print('[INFO] train and test split created...')
print(trainX.shape)
checkpoint = ModelCheckpoint('final_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
callbacks = [checkpoint]

model.fit(trainX, trainX, batch_size=64, epochs=2, validation_data=(testX, testX), callbacks=callbacks)  # model.save('first_try.h5')
# model.save('second_model_final.h5')
# print('[INFO] model saved...')
