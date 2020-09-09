#!/usr/bin/env python
# coding: utf-8

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt


#load the spectrograms
data_spec = np.load("/content/drive/My Drive/Deep Learning/Data_Spectrograms.pkl", allow_pickle = True)
x, y = data_spec


#split into testing and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)


#check distribution of classes
plt.hist(y_train)


#oversample the minority class using ADASYN
from imblearn.over_sampling import SMOTE , ADASYN
from imblearn.under_sampling import RandomUnderSampler

# These tools only work with 2D data, needed to reshape before using and reshaping again after it
X_train=X_train.reshape(X_train.shape[0],6000)

# create synthetic samples for the minority class
sm = ADASYN(sampling_strategy='minority')         
X_train, y_train = sm.fit_sample(X_train, y_train)



#check the new distribution of y_train, as well as the distribution of y_test
plt.hist(y_train)
print(y_train.shape)
plt.hist(y_test)
print(y_test.shape)


#reshape the spectrograms to fit the model architecture
X_train = X_train.reshape(X_train.shape[0], 30, 100, 2)
X_test = X_test.reshape(X_test.shape[0], 30, 100, 2)


#use to_categorical to create categorical vectors out of the integer targets
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#create and compile the model
from keras.layers import Dense, GRU, Input, LSTM, Bidirectional, Dropout, Activation, Conv1D, Conv2D, concatenate, SimpleRNN
from keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D, MaxPooling2D, AveragePooling1D, Flatten, Add, Reshape, RepeatVector
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics



def create_model(n_cnn_dense=64, drop=True, input_dim=2, timesteps=3000, fs=100, num_classes=6):
    
    input_dim = input_dim
    timesteps = timesteps   
    inp = Input(shape=(timesteps, input_dim))
    inp2 = Input(shape=(30,100,2))
    fs = fs
    num_classes = num_classes
    #84,57
    x = (Conv2D(filters=32, kernel_size=(5,5), activation='tanh', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.01)))(inp2)
    x = (Conv2D(filters=32, kernel_size=(3,3), activation='tanh',padding='valid'))(x)
    x = (Conv2D(filters=32, kernel_size=(3,3), activation='tanh',padding='valid'))(x)
    x = MaxPooling2D(pool_size=(2,2), strides=1, padding='same')(x)
    if drop:
        x = Dropout(0.2)(x)
    
    y = Conv2D(filters=64, kernel_size=(100,100), strides=1,  activation='tanh', padding='same', kernel_initializer = 'he_normal', kernel_regularizer = regularizers.l2(0.01))(inp2)
    y = (Conv2D(filters=32, kernel_size=(3,3), activation='tanh',padding='same'))(y)
    y = (Conv2D(filters=32, kernel_size=(3,3), activation='tanh',padding='same'))(y)
    y = (Conv2D(filters=32, kernel_size=(3,3), activation='tanh',padding='valid'))(y)
    y = (Conv2D(filters=128, kernel_size=(3,3), activation='tanh',padding='valid'))(y)
    y = (Conv2D(filters=32, kernel_size=(3,3), activation='tanh',padding='same'))(y)
    
    xy = concatenate([x, y], axis=3)        
    output = Dense(n_cnn_dense, activation='relu')(xy)
    pool_size = int(output.shape[1]) // 4
    output = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid')(output)
    output = Flatten()(output)
    output = RepeatVector(2)(output)
    output = SimpleRNN(100,activation='relu', kernel_initializer='glorot_uniform')(output)     
    output = Dense(num_classes, activation='softmax')(output)
    
    model = Model(inputs=inp2, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# Remove warnings
import logging
logging.getLogger('tensorflow').disabled = True 

#create model and model summary
model = create_model()
model.summary()


#fit the model on the train data, validate on test data
#use ModelCheckpoint to save the best performing model on validation accuracy
from keras.callbacks import ModelCheckpoint

mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True) 

model.fit(X_train, y_train,  validation_data=(X_test,y_test), verbose=1, batch_size=256, epochs= 50,callbacks= [mc])


# load the saved model
from keras.models import load_model
saved_model = load_model('best_model.h5')

# get training and testing accuracy
score = saved_model.evaluate(X_train, y_train, verbose=1)
print("Training Accuracy: ", score[1])
score = saved_model.evaluate(X_test, y_test, verbose=1)
print("Testing Accuracy: ", score[1])


#load testing set and reshape to right dimensions
test = np.load("/content/drive/My Drive/Deep Learning/Test_Spectrograms_no_labels.pkl", allow_pickle = True)
test = np.array(test)
test = test.reshape(1754, 30, 100, 2)

#generate predictions using our saved model
#use argmax to turn categorical vectors back into integer predictions
#save to a .txt file
predictions = saved_model.predict(test)
predictions = np.argmax(predictions, axis = 1)
np.savetxt("predictions19.txt", predictions, fmt='%i', encoding = None)

