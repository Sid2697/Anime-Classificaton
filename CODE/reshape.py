import cv2
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


path = '/Users/siddhantbansal/Desktop/Project_files/feb2ccb788aa34a70ad13199127c2923.jpg'
img = cv2.imread(path)
image = preprocess(img)
print('Shape of image is', image.shape)
cv2.imwrite('/Users/siddhantbansal/Desktop/img_5.jpg', image)
