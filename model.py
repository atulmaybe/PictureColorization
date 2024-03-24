from keras.layers import Conv2D, UpSampling2D, Input, Dense, BatchNormalizationV2
import tensorflow
from keras.models import Sequential
from keras.utils.image_utils import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator

from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave

import numpy as np
import tensorflow as tf

# image path
path = "train_data/"

# Normalize image pixel value because activation fun relu gores from o to 1 so we gonnn
# divide by 255 to rescale in between 0 to 1
train_image_generator = ImageDataGenerator(rescale=1./255)


train = train_image_generator.flow_from_directory(
    path,
    class_mode=None)


# Convert from RGB to LAB
X = []
Y = []


# Converting color image into Lab space
for img in train[0]:
    try:
        lab = rgb2lab(img)
        X.append(lab[:, :, 0])
        Y.append(lab[:, :, 1:]/128)
    except:
        print("Error")

X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,))  # to make the dimesion same as Y
print(X.shape)
print(Y.shape)


# encoder

model = Sequential()
model.add(Conv2D(64, (3, 3), activation="relu",
          padding='same', strides=2, input_shape=(256, 256, 1)))
model.add(Conv2D(128, (3, 3), activation="relu",
          padding='same'))
model.add(Conv2D(128, (3, 3), activation="relu",
          padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation="relu",
                 padding='same'))
model.add(Conv2D(256, (3, 3), activation="relu",
                 padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation="relu",
          padding='same'))
model.add(Conv2D(512, (3, 3), activation="relu",
          padding='same'))
model.add(Conv2D(512, (3, 3), activation="relu",
          padding='same'))
model.add(Conv2D(512, (3, 3), activation="relu",
          padding='same'))
model.add(Conv2D(256, (3, 3), activation="relu",
                 padding='same'))


# Decoder

model.add(Conv2D(128, (3, 3), activation="relu", padding='same'))

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu",
          padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation="relu",
          padding='same'))
model.add(Conv2D(16, (3, 3), activation="relu",
          padding='same'))
model.add(Conv2D(2, (3, 3), activation="tanh",
          padding='same'))
model.add(UpSampling2D((2, 2)))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()


model.fit(X, Y, validation_split=0.1, epochs=100, batch_size=5)

model.save('colorize.model')
