
import tensorflow as tf
from keras.utils.image_utils import img_to_array, load_img

from skimage.transform import resize
from skimage.io import imsave, imshow
import numpy as np
from skimage.color import rgb2lab, lab2rgb
###########################################################
# Load saved model and test on images.
# colorize_autoencoder300.model is trained for 300 epochs
#


def getColorImage(inputImage):

    model = tf.keras.models.load_model('colorize.model',
                                       custom_objects=None,
                                       compile=True)

    img1_color = []

    img1 = img_to_array(load_img(str(inputImage)))
    img1 = resize(img1, (256, 256))
    img1_color.append(img1)

    img1_color = np.array(img1_color, dtype=float)
    img1_color = rgb2lab(1.0/255*img1_color)[:, :, :, 0]
    img1_color = img1_color.reshape(img1_color.shape+(1,))

    output1 = model.predict(img1_color)
    output1 = output1*128

    result = np.zeros((256, 256, 3))
    result[:, :, 0] = img1_color[0][:, :, 0]
    result[:, :, 1:] = output1[0]
    imshow(lab2rgb(result))
    imsave("result/result.png", lab2rgb(result))
