# import libraries 
import streamlit as st
from PIL import Image, ImageOps
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing, Model, layers
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os


st.header("Image Class Predictor") # headed
st.write(""" Creator of App Hassan Jama""")
st.write(""" About the data: This data was initially published on https://datahack.analyticsvidhya.com by Intel to host a Image classification Challenge. You can find the dataset at this link https://www.kaggle.com/puneet6060/intel-image-classification

Content

This Data contains around 25k images of size 150x150 distributed under 6 categories(Building, Forest, Glacier, Mountain, Sea, Street)""")
st.write(""" Algorithm used: InceptionV3 with layers freezed during training so that the weights don't change during backpropagation.
Mixed_7 as the last layer of InceptionV3 pre-train model. Followed by adding a fully dense layer of 256, then a dropout of rate of .2, then a final dense layer of 6 for the 6 different classes with softmax as the activation function.
The model reach a validation accuracy of 91.6% and a validation loss of .3105 after 20 epochs of training. """)
# function were the image will be uploaded
def main():
  file_uploaded = st.file_uploader("Drop or drag images below", type = ["jpg", "png", "jpeg"]) # upload image
  if file_uploaded is not None:
    image = Image.open(file_uploaded) # open uploaded image
    figure = plt.figure() # plot figure
    plt.imshow(image) # than show ploted figure
    plt.axis("off") # axis is off
    result = predict_class(image) # predict class name
    st.write(result)
    st.pyplot(figure)


#function were the image will be predicted on using model
def predict_class(image):
  classifier_model = tf.keras.models.load_model(r'tl_model_tf.h5')
  shape = ((299,299,3))
  model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])
  test_image = image.resize((299,299))
  test_image = preprocessing.image.img_to_array(test_image)
  test_image = test_image/255.0
  test_image = np.expand_dims(test_image, axis = 0)
  class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
  predictions = model.predict(test_image)
  scores = tf.nn.softmax(predictions[0])
  scores = scores.numpy()
  image_class = class_names[np.argmax(scores)]
  result = 'The image uploaded is: {}'.format(image_class)
  return result



if __name__ == "__main__":
  main()
