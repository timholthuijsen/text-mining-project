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


# LSTM MODEL CODE IS ADAPTED FROM:
# https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


# --- CONFIGURE ---------------------------------------------------------------

DATANAME = "final_shortened"   # Name of the processed data file
N_RECIPES = 5                  # Number of sentences to generate when generate() is called
N_EPOCHS = 20                  # Number of epochs in model training
BATCH_SIZE = 128               # Bar-tch size while model training

# ----------------------------------------------------------------------------


def prepare_data(p=True):
    """
    Read the existing data
    p: bool = Be verbose
    Outputs a tuple of names, units and recipes
    """
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

def process_data(data, dumpname=DATANAME):
    """
    Clear and format the data
    """
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
    hp.dump(db, DATANAME)
    print(f"Processed data! Saved as {DATANAME}\n")
    return db


def init_vars():
    """
    Prepare the data for training
    """
    data = hp.dumpRead(DATANAME)
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
    """
    Create the LSTM model and save it as a .hdf5 file
    REQUIRES A FOLDER NAMED "model" IN THE DIRECTORY
    TAKES VERY LONG TO RUN
    """
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
    """
    Use the created model at "model/{fname}" to generate text
    Returns a recipe
    """
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
    """
    Uses create_recipe() to create a recipe
    Then selects the better ones
    """
    recipe = ["INGREDIENT"]
    def count_ingredients():
        n = 0
        for word in recipe:
            if "INGREDIENT" in word.upper():
                n += 1
        return n/len(recipe)
    attempts = 0
    # Make sure the sentence is decently long, and ingredients make up less than 40% of the sentence.
    while len(recipe) < 8 or count_ingredients() > 0.4:
        recipe = create_recipe(var=var)
        attempts += 1
        # Quit if recipe cannot be generated
        if attempts > 99:
            recipe = ["ERROR:", "", "", "", "Failed", "to", "generate", "a", "recipe", "\n"]
    # Remove common words that cannot end the sentence
    while recipe[-2] in ["to", "from", "is", "until", "and", "when", "an", "a", "or", "if", "on", "at"]:
        recipe = recipe[:-2] + recipe[-1:]
    # Generate ingredients
    ingredients = generate_ingredients()
    j = -1
    units = prepare_data(p=False)[1]
    # Put the generated iingredients in place
    for i, word in enumerate(recipe):
        if word == "INGREDIENT":
            j = min(j+1, len(ingredients)-1)
            recipe[i] = ingredients[j]
        elif word == "UNIT":
            recipe[i] = random.choice(list(units))
    return ".".join(" ".join(recipe).split(" \n"))

def check_contains(i, l, p=False):
    """
    Check if the item i (or singular versions) are in the list l
    """
    if p or len(i) > 3:
        return (i[-1] == "s" and i[:-1] in l) or (i[-2:] == "es" and i[:-2] in l) or (i[-3:] == "ies" and i[:-3] + "y" in l)
    return False

def read_coocs():
    """
    Read the file "Cococcs of ingredients"
    """
    raw = dict(hp.dumpRead("Cococcs of ingredients"))
    coocs = dict()
    for key in raw:
        coocs[(key[0][2:-1].lower(), key[1][2:-1].lower())] = raw[key]
    return dict(sorted(coocs.items(), key=lambda item: item[1], reverse=True))

def generate_ingredients_h(il=[], init=random.choice(hp.dumpRead("model/inglist_final"))):
    """
    Helper function for generate_ingredients()
    """
    coocs = read_coocs()
    selected = il
    for key in coocs:
        if len(il) < 20 and key[0] == init and key[0] not in selected:
            selected.append(key[1])
    return selected

def generate_ingredients():
    """
    Generate a list of related ingredients
    It is possible to comment out this code and add a list of ingredients manually.
    For example:
        selected = ["pasta", "tomato", "garlic", "oil", "onions", "salt", "pepper"]
        return selected
    This way you can use custom ingredients
    """
    first = ""
    selected = []
    # Choose a random, valid first ingredient
    while first == "" or len(first) > 30:
        first = random.choice(hp.dumpRead("model/inglist_final"))
    # Get a list of 20 ingredients that are related to the previous ones
    # Not all 20 are used, this is just to make sure that the model does not run out of ingredients.
    # 20 is deduced by:
    #   [maximum length of a sentence] x [maximum percentage of ingredients in a sentence]
    #   50 x 0.40 = 20
    while len(selected) < 20:
        selected += generate_ingredients_h(init=random.choice(hp.dumpRead("model/inglist_final")))
    return selected


# --- RUN ---------------------------------------------------------------------

# HOW TO USE
# 1. prepare()
# 2. create()
# 3. Update create_recipe() to use the desired .hdf5 file
# 4. generate()
    
# Prepare the data (skip if it is already processed)
def prepare(fname=DATANAME):
    raw_data = prepare_data()
    process_data(raw_data, fname)

# Create an LSTM model for text generation
def create(epochs=N_EPOCHS, batch_size=BATCH_SIZE):
    var = init_vars()
    create_model(var=var, epochs=epochs, batch_size=batch_size)

# Generate n sentences for the recipe
def generate(n=N_RECIPES):
    var = init_vars()
    for i in range(n):
        print(f"{i+1}/{n}:", select_recipe(var))
