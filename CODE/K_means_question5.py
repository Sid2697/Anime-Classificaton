# importing necessary packages
from preprocessing.preprocessing import AspectAwarePreprocessor
from keras.preprocessing.image import img_to_array
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from keras.models import Model
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import glob
import cv2
import os

# path to the survey CSV
data = pd.read_csv('/home/stu15/s15/ts6442/Capstone/codes/HIT_results.csv')
# path to the folder where labelled images are saved
path = '/home/stu15/s15/ts6442/Capstone/Labelled_images/'
# reading all the images from the file
image = glob.glob(path + '*.jpg')
print('[INFO] length of images', len(image))

images = []
labels = []
i = 0
sid = 0
# defining classes as dictionary
emotions = {'Action': 0, 'Sci-fi': 1, 'Fantasy': 2, 'Romance': 3, 'Horror': 4, 'Generic Drama': 5}
for item in image:
    # getting label corresponding to the image
    a = data['Answer.Q5Answer'].loc[data['Input.image_url'] == 'https://lijingyang.me/images/AmazonMTurk/' + item.split('/')[-1]]
    try:
        # removing more than one entry
        image = cv2.imread('/home/stu15/s15/ts6442/Capstone/Labelled_images/' + item.split('/')[-1])
        ap = AspectAwarePreprocessor(128, 128)
        image = ap.preprocess(image)
        image = img_to_array(image)
        b = a.values.tolist()[0].split('|')[0]
        images.append(image)
        labels.append(emotions[b])
        sid += 1
    except:
        # deleting files with label as "NaN"
        # os.remove('/home/stu15/s15/ts6442/Capstone/Labelled_images/' + item.split('/')[-1])
        i += 1
        print('[INFO] Exception at image number', sid)
        pass
    if sid % 500 == 0:
        # break
        print('[INFO] {} images loaded...'.format(sid))

images = np.array(images)
# labels = np.array(labels).reshape(len(labels), 1)
labels = np.array(labels)
print('[INFO] {} images not loaded'.format(i))
print('[INFO] shape of images is', images.shape)
print('[INFO] shape of labels is', labels.shape)

model_path = '/home/stu15/s15/ts6442/Capstone/codes/final_model.h5'
number = 4

model = load_model(model_path)
clip = Model(model.inputs, model.layers[number].output)
clip.summary()

encoded_images = []
a = 0
for item in images:
    item = item.reshape(1, item.shape[0], item.shape[1], item.shape[2])
    p = clip.predict(item)
    encoded_images.append(p[0])
    a += 1
    if a % 1000 == 0:
        print('[INFO] {} encoded images saved...'.format(a))

encoded_images = np.array(encoded_images)


encoded_images = encoded_images.reshape(encoded_images.shape[0], encoded_images.shape[1] * encoded_images.shape[2] * encoded_images.shape[3])
print('[INFO] shape of encoded images is', encoded_images.shape)
(trainX, testX, trainY, testY) = train_test_split(encoded_images, labels, test_size=0.2)
print('[INFO] train test split created...')


# print('[INFO] evaluating k-NN classifier...')
# model = KNeighborsClassifier(n_jobs=-1, algorithm='auto')
# model.fit(trainX, trainY)

print('[INFO] evaluating k-means classifier...')
model = KMeans(n_clusters=6, n_init=2, verbose=1)
model.fit(trainX, trainY)

print(classification_report(testY, model.predict(testX), target_names=['Action', 'Sci-fi', 'Fantasy', 'Romance', 'Horror', 'Generic Drama']))
