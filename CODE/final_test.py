# imporing necessary packages
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import argparse
from preprocessing.preprocessing import AspectAwarePreprocessor, ImageToArrayPreprocessor, SimplePreprocessor
from preprocessing.datasets import SimpleDatasetLoader
import cv2
import matplotlib.pyplot as plt
import imutils


def preprocess(image):
    '''
    grab the dimentions of the image and then initialize the deltas to use when cropping
    '''
    (h, w) = image.shape[:2]
    dW = 0
    dH = 0

    # if the width is smaller than the height, then resize along the width and then update the daltas to crop the height to the desired dimension
    if w < h:
        image = imutils.resize(image, width=128)
        dH = int((image.shape[0] - 128) / 2.0)

    # otherwise the height is smaller than the width so resize along the height and then update the deltas to crop along the width
    else:
        image = imutils.resize(image, height=128)
        dW = int((image.shape[1] - 128) / 2.0)

    # now that our image have been resized, we need to re-grab the width and height, followed by performing the crop
    (h, w) = image.shape[:2]
    image = image[dH:h - dH, dW:w - dW]

    # finally, resize the image to the provided spatial dimentions to ensure our output image is always a fixed size
    return cv2.resize(image, (128, 128))


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True, help='path to the test image')
ap.add_argument('-m', '--model', required=True, help='path to the model')
args = vars(ap.parse_args())

print('[INFO] loading model...')
model = load_model(args['model'])
model.summary()

print('[INFO] loading test image...')
image = cv2.imread(args['path'])
print('[INFO] type of image is', type(image))
# aap = AspectAwarePreprocessor(128, 128)
# iap = ImageToArrayPreprocessor()
# image = img_to_array(aap)
image = preprocess(image)
image = img_to_array(image)
# sdl = SimpleDatasetLoader(preprocessors=[aap, iap])

# (data, _) = sdl.load(args['path'], verbose=1)
data = image.astype('float') / 255.0

# print('[INFO] total number of images are ', len(data))
data = data.reshape(1, 128, 128, 3)
# print('Shape of the image is', data.shape)


p = model.predict(data)
# print('Type of output', type(p))
# print('Length of output', len(p))
# print(p)
cv2.imshow('Original Image', data.reshape(128, 128, 3))
cv2.waitKey(0)
# cv2.imshow('Reconstruction', p * 255.0)
# plt.imshow(p[0])
# print(p[0].shape)
cv2.imshow('Reconstruction', p[0])
cv2.imwrite('/Users/siddhantbansal/Desktop/Project_files/Siddhant_Code/models/4_layers/Reconstruction_' + args['path'].split('/')[-1], p[0] * 255.0)
cv2.waitKey(0)
