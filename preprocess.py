# -*- coding: utf-8 -*-

import nltk
from nltk.corpus import wordnet as wn

import helpers as hp


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
    
# Get a list of available databases
dbs = hp.getDatabases()

# Clean the first one
clean_db = hp.cleanFile(dbs[0])

# Print first 10 lines of database
hp.head(clean_db, 10)

# PoS-Tag, lemmatize and format
postagged = pos_tag_db(clean_db)
lemmatized = lemmatize_db(postagged, [".", "X"])
formatted = hp.formatFile(lemmatized)

# Print 5 lines from the result
hp.head(formatted)