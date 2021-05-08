# -*- coding: utf-8 -*-

import pandas as pd
import helpers as hp
import preprocess as pp
import string


# ADAPTED FROM: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


# Get ingredients and recipes
# data = pd.read_csv("data/nyt-ingredients-snapshot-2015.csv")
# names = [str(name).lower().strip(string.punctuation + " ") for name in data["name"]]
# names2 = pp.list_ingredients()
# names += names2
# units = {str(unit).lower().strip(string.punctuation + " ") for unit in data["unit"] if len(str(unit)) < 14 and not str(unit)[0].isdigit()}
# recipes = [row for row in hp.cleanFile(hp.getDatabases()[0]) if len(row) > 50 and not row[0].isdigit()]

# def prepare_data():
#     db = []
#     for row in recipes:
#         for word in row.split(" "):
#             word = word.strip().strip(string.punctuation + " ")
#             if len(word) > 0:
#                 if word in names:
#                     db.append("INGREDIENT")
#                 elif word in units:
#                     db.append("UNIT")
#                 else:
#                     db.append(word.lower())
#         db.append("\n")
#     return db

data = hp.dumpRead()

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

words = sorted(list(set(data)))
ewords = dict((c, i) for i, c in enumerate(words))
enumbs = dict((i, c) for i, c in enumerate(words))

seq_length = 100
dataX = []
dataY = []
for i in range(0, len(data) - seq_length, 1):
 	seq_in = data[i:i + seq_length]
 	seq_out = data[i + seq_length]
 	dataX.append([ewords[char] for char in seq_in])
 	dataY.append(ewords[seq_out])
n_patterns = len(dataX)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(len(words))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# model = Sequential()
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')
# # define the checkpoint
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# # fit the model
# model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

def create_recipe():
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], activation='softmax'))
    
    filename = "weights-improvement-20-3.2472.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    
    # generate characters
    recipe = []
    for i in range(50):
        if i == 0 or recipe[-1] != "\n":
            result = ""
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(len(words))
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = enumbs[index]
            # seq_in = [enumbs[value] for value in pattern]
            recipe.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
    return recipe

def select_recipe():
    recipe = ["INGREDIENT"]
    def count_ingredients():
        n = 0
        for word in recipe:
            if word in ["INGREDIENT", "INGREDIENT."]:
                n += 1
        return n/len(recipe)
    while len(recipe) < 8 or count_ingredients() > 0.5:
        recipe = create_recipe()
    while recipe[-2] in ["to", "from", "is", "until", "and", "when", "an", "a", "or", "if", "on", "at"]:
        recipe = recipe[:-2] + recipe[-1:]
    return ".".join(" ".join(recipe).split(" \n"))