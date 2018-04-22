import csv
import cv2
import numpy as np 
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

lines = [] 
images = [] 
measurements = [] 
correction = 0.15
EPOCHS = 3

with open('driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile) 
    for line in reader: 
        lines.append(line) 

with open('driving_log_1.csv') as csvfile: 
    reader = csv.reader(csvfile) 
    for line in reader: 
        lines.append(line) 

images = [] 
measurements = [] 
correction = 0.15
EPOCHS = 3

for line in lines:
    measurement = float(line[3])

    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'IMG/' + filename

        if(i == 0): 
            steer = measurement
        elif(i == 1): 
            steer = measurement + correction 
        else: 
            steer = measurement - correction 

        image = cv2.imread(current_path)
        images.append(image) 
        measurements.append(steer) 

        flipped = cv2.flip(image,1) 
        images.append(flipped) 
        measurements.append(steer*-1.0)


X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout 
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model 


model = Sequential() 
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320,3)))
model.add(Cropping2D(cropping =((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2),  activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
#model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
#model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
#model.add(Dropout(0.6))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=EPOCHS)

model.save('model.h5')
