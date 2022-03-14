import numpy as np
import pandas as pd
import tensorflow as tf
import os
import zipfile
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model, layers
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


local_zip = 'intel_image_class.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()
zip_ref.close()

base_dir = 'intel_image_class'

train_dir = os.path.join(base_dir, 'seg_train')
test_dir = os.path.join(base_dir, 'seg_test')

# Directory with our training pictures
train_buildings_dir = os.path.join(train_dir, 'buildings')
train_forest_dir = os.path.join(train_dir, 'forest')
train_glacier_dir = os.path.join(train_dir, 'glacier')
train_mountain_dir = os.path.join(train_dir, 'mountain')
train_sea_dir = os.path.join(train_dir, 'sea')
train_street_dir = os.path.join(train_dir, 'street')


# Directory with our testing pictures
test_buildings_dir = os.path.join(test_dir, 'buildings')
test_forest_dir = os.path.join(test_dir, 'forest')
test_glacier_dir = os.path.join(test_dir, 'glacier')
test_mountain_dir = os.path.join(test_dir, 'mountain')
test_sea_dir = os.path.join(test_dir, 'sea')
test_street_dir = os.path.join(test_dir, 'street')




# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )



# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'categorical',
                                                    target_size = (299, 299))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( test_dir,
                                                        batch_size = 20,
                                                        class_mode = 'categorical',
                                                        target_size = (299, 299))



iv3 = InceptionV3(input_shape = (299, 299, 3), include_top = False, weights = 'imagenet')


# Freeze the train layers of the InceptionV3 so that the weights don't change during backpropagation
for layer in iv3.layers:
  layer.trainable = False


last_layer = iv3.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
iv3_output = last_layer.output
iv3_output

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(iv3_output)
# Add a fully connected layers
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(6, activation='softmax')(x)

model2 = tf.keras.models.Model(iv3.input, x)

model2.compile(optimizer= RMSprop(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history_iv33 = model2.fit(train_generator, validation_data = validation_generator, epochs = 10)


model2.save('tl_model_tf.h5',save_format='tf')
