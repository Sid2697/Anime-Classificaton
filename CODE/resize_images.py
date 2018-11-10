from siddhantbansal.preprocessing import AspectAwarePreprocessor, ImageToArrayPreprocessor, SimplePreprocessor
from siddhantbansal.datasets import SimpleDatasetLoader
import argparse
import glob

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-s', '--size', type=int, default=128, help='desired size of the image')
args = vars(ap.parse_args())

# grab the list of images
print('[INFO] loading images...')
# imagePaths = glob.glob(args['dataset'])
imagePaths = glob.glob('Capstone/images/*.jpg')


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
print('Shape of data is', data.shape)
