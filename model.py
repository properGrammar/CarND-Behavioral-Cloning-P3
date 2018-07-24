# Udacity Self-Driving Car Nanodegree
# Project 3: Behavioral Cloning
# by Graham Arthur Mackenzie (graham.arthur.mackenzie@gmail.com)
import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img

# Read the text from our driving_log.csv into our samples array
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Note that, prior to using generators, our training / test split is done in
# model.fit(), whereas model.fit_generator() does not contain such a subfunction,
# which is why we're doing it here
# Training / testing split is the canonical 80 / 20
training_set, validation_set = train_test_split(samples, test_size=0.2)

# The generator() method definition
def generator(samples, batch_size=32):
    # The number of samples is equal to the length of the samples[] array
    num_samples = len(samples)
    # Loop forever so that the generator never terminates
    while 1:
        # Shuffle our samples
        shuffle(samples)
        # Iterate through a range of the total samples
        for offset in range(0, num_samples, batch_size):
            # Collect a batch of samples in sample_batch
            sample_batch = samples[offset:offset+batch_size]
            # Let's set up some arrays
            images = []
            angles = []
            augmented_images = []
            augmented_angles = []
            # Let's iterate through the samples in our batch
            for sample in sample_batch:
                # Assign the local root data path of the folder of images to data_path
                data_path = 'data/IMG/'
                # The zeroth element in sample[] is the data path for the center camera image
                center_path = sample[0]
                # Split on the '/' and then grab the last element of the resultant array
                center_filename = center_path.split('/')[-1]
                # Using cv2.imread() on data_path+center_filename gives us an actual image
                center_image = cv2.imread(data_path+center_filename)
                # We shrink (resize()) our image by half
                cv2.resize(center_image, None, None, 0.5, 0.5, interpolation = cv2.INTER_AREA)
                # Because cv2.imread() encodes in BGR color space, we have to convert the image to RGB
                # in order for drive.py to make best use of its results, insofar as drive.py uses RGB color space
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                # Repeat the above process for the left camera image
                left_path = sample[1]
                left_filename = left_path.split('/')[-1]
                left_image = cv2.imread(data_path+left_filename)
                cv2.resize(left_image, None, None, 0.5, 0.5, interpolation = cv2.INTER_AREA)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                # Repeat the above process for the right camera image
                right_path = sample[2]
                right_filename = right_path.split('/')[-1]
                right_image = cv2.imread(data_path+right_filename)
                cv2.resize(right_image, None, None, 0.5, 0.5, interpolation = cv2.INTER_AREA)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                # Append all three images to the images[] array
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                # The angle of the steering wheel (herein referred to as the center_angle)
                # is the third element in sample[]
                center_angle = float(sample[3])
                # We crib David Silver's suggested intuitive correction factor size of 0.2
                correction = 0.2
                # The left and right angles are the center_angle with correction added and taken away, respectively
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                # We append all three angles to the angles[] array
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
            # For the sake of augmenting our data set, 
            # let's zip together each image with its associated angle
            for image,angle in zip(images, angles):
                # And, first of all, add each image in images[] to augmented_images[]...
                augmented_images.append(image)
                # then add each angle in angles[] to augmented_angles[]...
                augmented_angles.append(angle)
                # then add each image in images[]  to augmented_images[], after being flipped horizontally...
                augmented_images.append(cv2.flip(image,1))
                # and, lastly, add each angle in angles[] to augmented_angles[] after multiplying it by -1.0 
                # (such that it accurately depicts the steering angle that would go with the image transformed above)
                augmented_angles.append(angle * (-1.0))
            # Let's turn our images and angles into numpy arrays
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            # And lastly yield a shuffled version of each of the above
            yield shuffle(X_train, y_train)

# Compile and train the model using the generator() function
training_generator = generator(training_set, batch_size=32)
validation_generator = generator(validation_set, batch_size=32)

# The Nvidia model
model = Sequential()
# Here's our preprocessing Lambda layer, which normalizes and mean-centers the data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# Our initial Cropping2D() layer:
# model.add(Cropping2D(cropping=((70, 25),(0, 0))))
# We reduce the dimensions of the above 2D Crop by 1/2, since we are
# reducing the dimensions of the images by the same ratio in generator() above
model.add(Cropping2D(cropping=((35, 12),(0, 0))))
# 5 Convolution layers, some with strides
# These contain our model's nonlinearity in the form of the 'relu' activation
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
# Flattening layer
model.add(Flatten())
# 4 Fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Here's the Adam optimizer we use, such that manually training the learning rate isn't necessary
model.compile(loss='mse', optimizer='adam')
# We double len(training_set), and thereby samples_per_epoch, below
# in order to account for the doubling of the dataset via augmentation done in generator()
# Note: our model achieves results in 3 Epochs
model.fit_generator(training_generator, samples_per_epoch= (len(training_set) * 2), validation_data=validation_generator, nb_val_samples=len(validation_set), nb_epoch=3)
# Remember to save your work! : )
model.save('model.h5')
# Thanks For Taking the Time to Read & Review My Code! and Be Well : )
exit()
