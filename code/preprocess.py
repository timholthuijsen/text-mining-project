# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import helpers as hp
import csv


MAP = {"VERB" : wn.VERB, "NOUN" : wn.NOUN, "ADJ" : wn.ADJ, "ADV" : wn.ADV}


def pos_tag_db(db: list) -> list:
    """
    PoS-Tag a given database

    Parameters
    ----------
    db : list
        Database to PoS-Tag.

    Returns
    -------
    list
        PoS-Tagged database.

    """
    return [nltk.pos_tag(sentence.split(), tagset = "universal") for sentence in db]
        

def lemmatize_db(db: list, exclude: list = []) -> list:
    """
    Lemmatize given database

    Parameters
    ----------
    db : list
        PoS-Tagged dataset to lemmatize.
    exclude : list, optional
        Types to exclude from the output. The default is [].

    Returns
    -------
    list
        Lemmatized database.

    """
    lemmatized_db = []
    for sentence in db:
        lemmatized_sentence = []
        for w, p in sentence:
            if p in exclude:
                continue
            elif p in MAP.keys():
                lemma = nltk.WordNetLemmatizer().lemmatize(w, pos = MAP[p])
            else:
                lemma = nltk.WordNetLemmatizer().lemmatize(w)
            lemmatized_sentence.append((lemma, p))
        lemmatized_db.append(lemmatized_sentence)
    return lemmatized_db
    

# EXAMPLE USE:
    
# # Get a list of available databases
# dbs = hp.getDatabases()

# # Clean the first one
# clean_db = hp.cleanFile(dbs[0])

# # Print first 10 lines of database
# hp.head(clean_db, 10)

# # PoS-Tag, lemmatize and format
# postagged = pos_tag_db(clean_db)
# lemmatized = lemmatize_db(postagged, [".", "X"])
# formatted = hp.formatFile(lemmatized)

# # Print 5 lines from the result
# hp.head(formatted)

# extracting ingredients 


def list_ingredients():
    # read json files
    df1 = pd.read_json("data/train.json")
    df2 = pd.read_json("data/test.json")
    
    # extract ingredients coloumn and convert to one list of ingredients
    df1_ingre = df1["ingredients"]
    df2_ingre = df2["ingredients"]
    
    all_ingre = pd.Series.tolist(df1_ingre) + pd.Series.tolist(df2_ingre)
    
    # convert list of lists to a flat list
    list_of_ingre = []
    
    for element in all_ingre:
        for item in element:
            list_of_ingre.append(item)
    
    # remove duplicates        
    final_ingre = list(dict.fromkeys(list_of_ingre)) # list of 7137 ingredients
    
    with open('final_ingr.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(final_ingre)
    
    return final_ingre
        