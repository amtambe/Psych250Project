!pip install --upgrade tensorflow

import tensorflow as tf
#only if we want to execute eagerly. Keep it at the top of the code.
#tf.enable_eager_execution() 
from tensorflow import keras
import sys
import copy
import queue

sys.modules['keras'] = keras

# ipython configuration
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

!pip install -q pyyaml  # Required to save models in YAML format

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np

# standard library
!pip install tensorflow-datasets
import tensorflow_datasets as tfds

#Import the retina and other libraries
!pip install git+git://github.com/dicarlolab/retinawarp
import matplotlib.pyplot as plt
from retina.retina_tf import warp_image

!pip install tensorflow-transform
import cv2
from tensorflow_transform import beam


EPOCHS = 10
OUTPUT_CLASSES = 10
#I ended up using a subset of the data because I had to
#train locally, and couldn't iterate over all the data
#quickly enough to train the model in time. 
DATA_SIZE = 960 
STEPS_PER_EPOCH = DATA_SIZE/32

dummy = tf.data.Dataset.from_tensors(0).repeat(None)

#We will first create 2 empty models and will later train them side by side
#This is just AlexNet.
def make_model():
  model = tf.keras.Sequential()
  
  # 1st Convolutional Layer
  model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(4,4), padding="valid"))
  model.add(Activation("relu"))
  # Max Pooling
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

  # 2nd Convolutional Layer
  model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding="valid"))
  model.add(Activation("relu"))
  # Max Pooling
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

  # 3rd Convolutional Layer
  model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
  model.add(Activation("relu"))

  # 4th Convolutional Layer
  model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding="valid"))
  model.add(Activation("relu"))

  # 5th Convolutional Layer
  model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="valid"))
  model.add(Activation("relu"))
  # Max Pooling
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))

  # Passing it to a Fully Connected layer
  model.add(Flatten())
  # 1st Fully Connected Layer
  model.add(Dense(4096, input_shape=(224*224*3,)))
  model.add(Activation("relu"))
  # Add Dropout to prevent overfitting
  model.add(Dropout(0.4))

  # 2nd Fully Connected Layer
  model.add(Dense(4096))
  model.add(Activation("relu"))
  # Add Dropout
  model.add(Dropout(0.4))

  # 3rd Fully Connected Layer
  model.add(Dense(OUTPUT_CLASSES))
  model.add(Activation("relu"))
  # Add Dropout
  model.add(Dropout(0.4))

  # Output Layer
  model.add(Dense(OUTPUT_CLASSES))
  model.add(Activation("softmax"))

  return model

#Create our two models and print the architecture
blur_model = make_model()
blur_model.summary()

fisheye_model = make_model()
fisheye_model.summary()

#We're going to use Cifar10 for reasons of speed
data, info = tfds.load("cifar10", with_info=True)

#If we want to use ImageNet instead, uncomment this.
#data, info = tfds.load("imagenet2012", with_info=True)
#train_data, test_data = data['train'], data['test']
#assert isinstance(train_data, tf.data.Dataset)
#assert info.features['label'].num_classes == 1000


#First, we're going to define the transformations used
# to filter our data. We apply the transformations to 
#individual tensors themselves, and then to each
#piece of data (which is just a dict) in the Dataset
#object.

#Transforms a single Tensor by retina warp. See citation at bottom.
def fisheye_transform_tensor(tensor):
  if tf.executing_eagerly():
    image = cv2.resize(tensor.numpy(), (224, 224))
    return warp_image(image, output_size=224)
  else:
    image = cv2.resize(tf.Session().run(tensor), (224, 224))
    return warp_image(image, output_size=224)

#Transforms a single piece of data by retina warp. 
def fisheye_transform_data(data):
  new = copy.copy(data)
  tensor = fisheye_transform_tensor(data["image"])
  new["image"] = tensor
  return new

#Executes uniform Gaussian blur on a single tensor.
def blur_transform_tensor(tensor):
  if tf.executing_eagerly():
    image = cv2.resize(tensor.numpy(), (224, 224))
    return cv2.blur(image, (5,5))
  else:
    image = cv2.resize(tf.Session().run(tensor), (224, 224))
    #on images of size 224x224, 4x4 is close to the mean blur
    return cv2.blur(image, (4,4)) 

#Executes uniform Gaussian blur on a single tensor
def blur_data(data):
  new = copy.copy(data)
  blur_tensor = blur_transform_tensor(data["image"])
  new["image"] = blur_tensor
  return new

#Now, split the data into test and train data sets. 
train_data, test_data = data['train'], data['test']
assert isinstance(train_data, tf.data.Dataset)

#Loop through the train data once, and the test data once
#Transform each element, enqueue it, and later we will 
#construct a new dataset by dequeueing from that queue. 
#This is because Datasets are not mutable in Tensorflow.
#But it is still advantageous to use them because it makes
#training stage much faster with Tensorflow optimizations.
q_fisheye_train = queue.Queue(DATA_SIZE + 1)
q_blur_train = queue.Queue(DATA_SIZE + 1)
train_iter = train_data.make_one_shot_iterator()
train_element = train_iter.get_next()
for i in range(0, DATA_SIZE):
  if i % 10 == 0:
    print(i)
  fisheye_element = fisheye_transform_data(train_element)
  blur_element = blur_data(train_element)
  q_fisheye_train.put(fisheye_element)
  q_blur_train.put(blur_element)
  train_element = train_iter.get_next()

#Repeat the above for the test data
q_fisheye_test = queue.Queue(DATA_SIZE / 4) #Keeps a 80/20 split of training to test data
q_blur_test = queue.Queue(DATA_SIZE / 4)
test_iter = test_data.make_one_shot_iterator()
test_element = test_iter.get_next()
for i in range(0, 25):
  fisheye_element = fisheye_transform_data(test_element)
  blur_element = blur_data(test_element)
  q_fisheye_test.put(fisheye_element)
  q_blur_test.put(blur_element)
  test_element = test_iter.get_next()

# Create datasets by dequeuing successive elements from `q`.
fisheye_train = dummy.map(lambda _: q_fisheye_train.get())
blur_train = dummy.map(lambda _: q_blur_train.get())
fisheye_test = dummy.map(lambda _: q_fisheye_test.get())
blur_test = dummy.map(lambda _: q_blur_test.get())

#Commented out for now, but this code will display the 
#first image of each dataset to visualize the transforms
#done on them.
#fig=plt.figure(figsize=(10, 10))
#fig.add_subplot(1, 2, 1)

#sess = tf.Session()
#blur_iter = blur_train.make_initializable_iterator()
#sess.run(blur_iter.initializer)
#blur_tensor = blur_iter.get_next()["image"]
#image = sess.run(blur_tensor)
#plt.imshow(image)


#fig.add_subplot(1, 2, 2)
#fisheye_iter = fisheye_train.make_initializable_iterator()
#sess.run(fisheye_iter.initializer)
#trans_tensor = fisheye_iter.get_next()["image"]
#trans_image = sess.run(trans_tensor)
#plt.imshow(trans_image)

#plt.show()

#Now, we begin the training. First, compile the two models. 
blur_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
fisheye_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

sess = tf.Session()

#Put the data into batches (this is a TensorFlow optimization)
blur_iter_batch = blur_test.batch(32).make_initializable_iterator()
fisheye_iter_batch = fisheye_test.batch(32).make_initializable_iterator()

#Define two iterators to iterate through the blur and fisheye data
#which we will use to feed data into the training algorithm

#Iterate through the uniform blur data set, yielding images with lables
def blur_iterator():
    sess.run(blur_iter_batch.initializer) #re-initialize the iterator

    while True:
        next_val = sess.run(blur_iter_batch.get_next())
        yield next_val['image'], next_val['label'],

#Iterate through the retina warped data set, yielding images with lables
def fisheye_iterator():
    sess.run(fisheye_iter_batch.initializer) #re-initialize the iterator

    while True:
        next_val = sess.run(fisheye_iter_batch.get_next())
        yield next_val['image'], next_val['label'],

#Call the tensorflow iterator to feed data for however many epochs
#This method will automatically create a split (1/8) of validation data
print ("Train the blur model")
blur_model.fit(blur_iterator(), epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

print ("Train the fisheye model")
fisheye_model.fit(fisheye_iterator(), epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

#Now, test the model - first, we will batch the data
blur_iter_test = blur_train.batch(32).make_initializable_iterator()
fisheye_iter_test = fisheye_train.batch(32).make_initializable_iterator()

#As above, we define two iterators through the test data
#Iterates through the uniform blur test data and yields images with labels
def blur_test_iterator():
    sess.run(blur_iter_test.initializer) #re-initialize the iterator

    while True:
        next_val = sess.run(blur_iter_test.get_next())
        yield next_val['image'], next_val['label'],

#Iterates through the fisheye blur test data and yields images with labels
def fisheye_test_iterator():
    sess.run(fisheye_iter_test.initializer) #re-initialize the iterator

    while True:
        next_val = sess.run(fisheye_iter_test.get_next())
        yield next_val['image'], next_val['label'],


#Compute the loss on each model. The method will also print the percent
# accuracy for each model alongside the actual loss. 
print("Evaluate the blur model:")
blur_loss = blur_model.evaluate(blur_test_iterator(), steps=1)
print(blur_loss)

print("Evaluate the fisheye model:")
fisheye_loss = fisheye_model.evaluate(fisheye_test_iterator(), steps=1)
print(fisheye_loss)


#@article{bashivan2018neural,
#  title={Neural population control via deep image synthesis},
#  author={Bashivan, Pouya and Kar, Kohitij and DiCarlo, James},
#  journal={bioRxiv},
#  pages={461525},
#  year={2018},
#  publisher={Cold Spring Harbor Laboratory}
#}
