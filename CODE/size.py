max1 = 0
max2 = 0
for file in glob.glob('images/*.jpg'):
    image = cv2.imread(file)
    sh = image.shape
    a = sh[0]
    b = sh[1]
    if a > max1:
        max1 = a
    if b > max2:
        max2 = b
