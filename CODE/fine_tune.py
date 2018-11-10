# importing necessary packages
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import cv2

# loading pretrained autoencoder


def haha():
    model = load_model('/Users/siddhantbansal/Desktop/Project_files/Siddhant_Code/Final_Results/5_layers_model/third_and_final_batch.h5')
    # print('==========================================')
    # model.summary()
    # print('==========================================')

    # clipping off half of the model
    # print('[INFO] clipping model...')
    clip = Model(model.inputs, model.layers[-6].output)
    # print('==========================================')
    # clip.summary()
    # print('==========================================')

    # defining a new model
    new = Sequential()
    new.add(clip)
    new.add(Dense(500, activation='relu'))
    new.add(Flatten())
    new.add(Dense(300, activation='relu'))
    new.add(Dense(100, activation='relu'))
    new.add(Dense(7, activation='softmax'))
    # print('==========================================')
    # print('[INFO] final model...')
    # print('==========================================')
    new.summary()

    # selecting which layers to train
    for layer in new.layers:
        layer.trainable = False
        break

    # checking which layers are getting trained and which are not
    for layer in new.layers:
        print(layer, layer.trainable)

    return new


model = haha()
model.summary()


# image = cv2.imread('/Users/siddhantbansal/Desktop/Project_files/fea4203b3cf1c6bc86c956383f2399c0.jpg')
# image = cv2.resize(image, (128, 128))
# image = image.reshape(1, 128, 128, 3)

# p = new.predict(image)
# plt.imshow(p[0][1])
# plt.savefig('/Users/siddhantbansal/Desktop/trail.pdf')

# new.add(Dense(500, activation='relu'))
# new.add(Dense(2), activation='softmax')
