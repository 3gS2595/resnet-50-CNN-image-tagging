import tensorflow as tf
import cv2
import os
from keras.utils import load_img, img_to_array
import numpy as np
import glob

IMAGE_SIZE = 224

mydir = "C:\\Users\\lucious\\Pictures\\selfTest"
file_list = glob.glob(mydir + "/*")  # Include slash or it will search in the wrong directory!!
print('file_list {}'.format(file_list))
print(len(file_list))

trainMyImagesFolder = "C:\\Users\\lucious\\Pictures\\train0"
CLASSES = os.listdir(trainMyImagesFolder)
num_classes = len(CLASSES)

best_model_file = 'C:\\Users\\lucious\\Desktop\\RESNET\\model0.h5'
model = tf.keras.models.load_model(best_model_file)


def prepareImage(images):
    ret = []
    for prepImg in images:
        print(prepImg)
        image = load_img(prepImg, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        imgResult = img_to_array(image)
        imgResult = np.expand_dims(imgResult, axis=0)
        imgResult = imgResult / 255.
        ret.append(imgResult)
    return ret


img = [cv2.imread(img) for img in file_list]
batch_images = prepareImage(file_list)

for x in range(0, len(batch_images)):
    predictions = model.predict(batch_images[x])
    print(predictions)

    answer = np.argmax(predictions, axis=1)
    print(answer)

    index = answer[0]
    className = CLASSES[index]

    print("The predicted class is : " + className)

    cv2.putText(img[x], className, (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("img", img[x])
    cv2.waitKey(0)
