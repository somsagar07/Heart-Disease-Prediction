# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 00:21:51 2021

@author: Sagar
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
'''
data = pd.read_csv('heart.csv')
dt = data.iloc[:,:13].values
target = data["target"].values
#print(target)
xtrain, xtest, ytrain ,ytest = train_test_split(dt, target, test_size=0.3, stratify=target , random_state=2)
#print(xtrain)
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(13, activation="relu"))
model.add(tf.keras.layers.Dense(14, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(optimizer='adam' , loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(xtrain, ytrain, epochs=100)
model.save('model')
'''
model = tf.keras.models.load_model('model')

inp=(63,1,3,145,233,1,0,150,0,2.3,0,0,1)

ainp=np.asarray(inp)
ab=ainp.reshape(1,-1)
prediction = model.predict(ab)

if int(prediction) == 1:
    print('heart disease detected')
else:
    print('safe')