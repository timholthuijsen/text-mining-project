# -*- coding: utf-8 -*-

import pandas as pd
import helpers as hp
import preprocess as pp
import string
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import random


# ADAPTED FROM: https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


# --- CONFIGURE ---------------------------------------------------------------

WEIGHTNAME = "final_shortened"
N_RECIPES  = 5

N_EPOCHS    = 20
BATCH_SIZE  = 128

# ----------------------------------------------------------------------------


def prepare_data(p=True):
    if p: print("Preparing data...")
    data = pd.read_csv("data/nyt-ingredients-snapshot-2015.csv")
    names = [str(name).lower().strip(string.punctuation + " ") for name in data["name"]] + pp.list_ingredients()
    units = {str(unit).lower().strip(string.punctuation + " ") for unit in data["unit"] if len(str(unit)) < 14 and not str(unit)[0].isdigit()}
    databases = hp.getDatabases()
    recipes = []
    for i in range(len(databases)):
        recipes += [row for row in hp.cleanFile(databases[i]) if len(row) > 50 and not row[0].isdigit()]
        if p: print(f"Loaded database {i+1}/{len(databases)}")
    if p: print("Prepared data!\n")
    return (names, units, recipes)

def process_data(data, dumpname=WEIGHTNAME):
    print("Processing data...")
    names, units, recipes = data
    db = []
    len_recipes = len(recipes)
    for i, row in enumerate(recipes):
        if i%100 == 0: print(f"Processing data... {i}/{len_recipes}")
        for word in row.split(" "):
            word = word.strip().strip(string.punctuation + " ")
            if len(word) > 0:
                if word in names or check_contains(word, names):
                    db.append("INGREDIENT")
                else:
                    db.append(word.lower())
        db.append("\n")
    hp.dump(db, WEIGHTNAME)
    print(f"Processed data! Saved as {WEIGHTNAME}\n")
    return db


def init_vars():
    data = hp.dumpRead(WEIGHTNAME)
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
    X = X / float(len(words))
    y = np_utils.to_categorical(dataY)
    return (X, y, dataX, dataY, words, ewords, enumbs)

def create_model(var, epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    X, y, dataX, dataY, words, ewords, enumbs = var
    print("Creating model...")
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    filepath="model/weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    print("Created model!\n")
    return model

def create_recipe(var, fname = "weights-20-3.1225.hdf5"):
    X, y, dataX, dataY, words, ewords, enumbs = var
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], activation='softmax'))
    filename = "model/" + fname
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    start = np.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    
    recipe = []
    for i in range(50):
        if i == 0 or recipe[-1] != "\n":
            result = ""
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(len(words))
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = enumbs[index]
            recipe.append(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
    # print(recipe)
    return recipe

def select_recipe(var):
    recipe = ["INGREDIENT"]
    def count_ingredients():
        n = 0
        for word in recipe:
            if "INGREDIENT" in word.upper():
                n += 1
        return n/len(recipe)
    attempts = 0
    while len(recipe) < 8 or count_ingredients() > 0.4:
        recipe = create_recipe(var=var)
        attempts += 1
        if attempts > 99:
            recipe = ["ERROR:", "", "", "", "Failed", "to", "generate", "a", "recipe", "\n"]
    while recipe[-2] in ["to", "from", "is", "until", "and", "when", "an", "a", "or", "if", "on", "at"]:
        recipe = recipe[:-2] + recipe[-1:]
    ingredients = generate_ingredients()
    j = -1
    units = prepare_data(p=False)[1]
    for i, word in enumerate(recipe):
        if word == "INGREDIENT":
            j = min(j+1, len(ingredients)-1)
            recipe[i] = ingredients[j]
        elif word == "UNIT":
            recipe[i] = random.choice(list(units))
    return ".".join(" ".join(recipe).split(" \n"))

def check_contains(i, l, p=False):
    if p or len(i) > 3:
        return (i[-1] == "s" and i[:-1] in l) or (i[-2:] == "es" and i[:-2] in l) or (i[-3:] == "ies" and i[:-3] + "y" in l)
    return False

def read_coocs():
    raw = dict(hp.dumpRead("Cococcs of ingredients"))
    coocs = dict()
    for key in raw:
        coocs[(key[0][2:-1].lower(), key[1][2:-1].lower())] = raw[key]
    return dict(sorted(coocs.items(), key=lambda item: item[1], reverse=True))

def generate_ingredients_h(il=[], init=random.choice(hp.dumpRead("model/inglist_final"))):
    coocs = read_coocs()
    selected = il
    for key in coocs:
        if len(il) < 20 and key[0] == init and key[0] not in selected:
            selected.append(key[1])
    return selected

def generate_ingredients():
    first = ""
    selected = []
    while first == "" or len(first) > 30:
        first = random.choice(hp.dumpRead("model/inglist_final"))
    while len(selected) < 20:
        selected += generate_ingredients_h(init=random.choice(hp.dumpRead("model/inglist_final")))
    return selected


# --- RUN ---------------------------------------------------------------------
    
def prepare(fname=WEIGHTNAME):
    raw_data = prepare_data()
    process_data(raw_data, fname)

def create(epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    var = init_vars()
    create_model(var=var, epochs=epochs, batch_size=batch_size)

def generate(n=N_RECIPES):
    var = init_vars()
    for i in range(n):
        print(f"{i+1}/{n}:", select_recipe(var))