# importing necessary packages
from preprocessing.preprocessing import AspectAwarePreprocessor
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import cv2
import os

# path to the survey CSV
data = pd.read_csv('/home/stu15/s15/ts6442/Capstone/codes/HIT_results.csv')
path = '/home/stu15/s15/ts6442/Capstone/faces/'
# reading all the images from the file
image = glob.glob(path + '*.jpg')
print('[INFO] length of images', len(image))

images = []
labels = []
i = 0
sid = 0
# defining classes as dictionary
emotions = {'Happy': 0, 'Sad': 1, 'Anger': 2, 'Neutral': 3, 'Surprise': 4, 'Disgust': 5, 'Scared': 6, 'Not clear': 7}

for item in image:
    # getting label corresponding to the image
    a = data['Answer.Q7Answer'].loc[data['Input.image_url'] == 'https://lijingyang.me/images/AmazonMTurk/' + item.split('/')[-1]]
    try:
        # removing more than one entry
        image = cv2.imread('/home/stu15/s15/ts6442/Capstone/faces/' + item.split('/')[-1])
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
        print('[INFO] {} images loaded...'.format(sid))

images = np.array(images)
labels = np.array(labels).reshape(len(labels), 1)
print('[INFO] {} images not loaded'.format(i))
print('[INFO] shape of images is', images.shape)
print('[INFO] shape of labels is', labels.shape)

(trainX, testX, trainY, testY) = train_test_split(images, labels, test_size=0.2)

model_path = '/home/stu15/s15/ts6442/Capstone/codes/final_model.h5'
number = 4


def fineTune(model, num):
    model = load_model(model)
    clip = Model(model.inputs, model.layers[num].output)

    # defining a new model
    new = Sequential()
    new.add(clip)
    # new.add(Dense(500, activation='relu'))
    new.add(Conv2D(256, (3, 3)))
    new.add(Activation('relu'))
    new.add(BatchNormalization())
    new.add(Dropout(0.5))
    # new.add(Conv2D(128, (3, 3)))
    # new.add(Activation('relu'))
    # new.add(BatchNormalization())
    # new.add(MaxPooling2D(pool_size=(2, 2)))
    # new.add(Dropout(0.5))
    # new.add(Conv2D(64, (3, 3), padding='same'))
    # new.add(Activation('relu'))
    # new.add(BatchNormalization())
    # new.add(Dropout(0.5))
    # new.add(Conv2D(128, (3, 3), padding='same'))
    # new.add(Activation('relu'))
    # new.add(BatchNormalization())
    # new.add(MaxPooling2D(pool_size=(2, 2)))
    # new.add(Dropout(0.5))
    # new.add(Conv2D(256, (3, 3), padding='same'))
    # new.add(Activation('relu'))
    # new.add(BatchNormalization())
    # new.add(Dropout(0.5))
    # new.add(Conv2D(512, (3, 3), padding='same'))
    # new.add(Activation('relu'))
    # new.add(BatchNormalization())
    # new.add(MaxPooling2D(pool_size=(2, 2)))
    # new.add(Dropout(0.5))
    new.add(Flatten())
    new.add(Dense(500, activation='relu'))
    new.add(Dropout(0.25))
    new.add(Dense(300, activation='relu'))
    new.add(Dropout(0.25))
    new.add(Dense(100, activation='relu'))
    new.add(Dropout(0.25))
    new.add(Dense(8, activation='softmax'))

    # selecting which layers to train
    # for layer in new.layers:
    # layer.trainable = False
    # break

    return new


def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf

    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)


print('[INFO] creating model...')
model = fineTune(model_path, number)
model.summary()

trainY = to_categorical(trainY)
testY = to_categorical(testY)

# opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall])

H = model.fit(trainX, trainY, epochs=40, verbose=1, validation_data=(testX, testY))
p = model.predict(testX)
for item in p[0]:
    print(item.argmax())
