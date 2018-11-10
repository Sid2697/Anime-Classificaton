# importing necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.preprocessing import AspectAwarePreprocessor, ImageToArrayPreprocessor, SimplePreprocessor
from preprocessing.datasets import SimpleDatasetLoader
from keras.layers import Input, Dense, Flatten, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import SGD
from imutils import paths
import numpy as np
import argparse
import glob
import cv2
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--size', type=int, default=128, help='desired size of the image')
args = vars(ap.parse_args())

# grab the list of images
print('[INFO] loading images...')
imagePaths = glob.glob('/home/stu15/s15/ts6442/Capstone/images/images/*.jpg')

# Resize the image keeping aspect ratio in mind
aap = AspectAwarePreprocessor(args['size'], args['size'])
# Resize the image without aspect ratio in mind
# sp = SimplePreprocessor(128, 128)
# converting images to array for easier processing
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
# as there are no labels using '_' in place of labels
(data, _) = sdl.load(imagePaths, verbose=1000)
data = data.astype('float') / 255.0
print('[INFO] total number of images are ', len(imagePaths))


(trainX, testX, _, _) = train_test_split(data, _, test_size=0.1)
print('[INFO] train and test split created...')
print(trainX.shape)

input_layer = Input(shape=(args['size'], args['size'], 3))
x = Conv2D(128, 5, activation='relu')(input_layer)
x = MaxPool2D(2)(x)
x = Conv2D(256, 2, activation='relu')(x)
x = MaxPool2D(2)(x)
# x = Conv2D(30, 2, activation='relu')(x)
# x = MaxPool2D(2)(x)
encoded = x
# x = UpSampling2D(2)(x)
# x = Conv2DTranspose(30, 2, activation='relu')(x)
x = UpSampling2D(2)(x)
x = Conv2DTranspose(256, 2, activation='relu')(x)
x = UpSampling2D(2)(x)
x = Conv2DTranspose(128, 5, activation='relu')(x)
x = Conv2DTranspose(3, 3, activation='sigmoid')(x)


model = Model(input=input_layer, output=x)
model.summary()


# opt = SGD(lr=0.001, decay=0.05 / 10, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint('first_batch.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
callbacks = [checkpoint]

model.fit(trainX, trainX, batch_size=64, epochs=30, validation_data=(testX, testX), callbacks=callbacks)

# model.save('first_try.h5')
model.save('first_batch.h5')
print('[INFO] model saved...')

testImg = testX[0]

prediction = model.predict(testX)[0]

cv2.imshow('Original', testImg)
cv2.imshow('Prediction', prediction)
# save the output image so that I can see it
# save 4-5 images for the report
# plot the training loss and accuracy
plt.style.use('ggplot')
fig = plt.figure()
plt.plot(np.arange(0, 30), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 30), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 30), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 30), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
fig.savefig('first_batch_image.pdf')
plt.show()
