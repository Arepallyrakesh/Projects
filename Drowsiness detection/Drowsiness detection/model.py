import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model
from keras.utils import plot_model
# from keras.utils.vis_utils import plot_model

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 30
TS=(24,24)
train_batch= generator('E:/projects main/Drowsiness detection/Drowsiness detection/data/train',shuffle=True, batch_size=BS,target_size=TS)
test_batch= generator('E:/projects main/Drowsiness detection/Drowsiness detection/data/test',shuffle=True, batch_size=BS,target_size=TS)
SPE = len(train_batch.classes)//BS
VS = len(test_batch.classes)//BS
print(SPE,VS)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3),padding="same", activation='relu', input_shape=(24,24,1)),

    MaxPooling2D(pool_size=(2,2),padding="same"),

    Conv2D(32,(3,3),padding="same",activation='relu'),
    MaxPooling2D(pool_size=(2,2),padding="same"),
    Conv2D(64, (3, 3),padding="same", activation='relu'),
    MaxPooling2D(pool_size=(2,2),padding="same"),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batch, validation_data=test_batch, epochs=10, steps_per_epoch=SPE, validation_steps=VS)
model.save('models/cnnCat2.h5', overwrite=True)
model.summary()