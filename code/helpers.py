import glob
import re
import pickle
import string
import pandas as pd

def getDatabases() -> list:
    """
    Get the list of available datasets
    
    Returns
    -------
    LIST
        Datasets in the /data/* directory.

    """
    return glob.glob("data/*.exl")

def cleanFile(filename: str) -> list:
    """
    Cleans the data from empty lines and metadata.

    Parameters
    ----------
    filename : str
        Path to the dataset file.

    Returns
    -------
    list
        Cleaned list of lines from the dataset.

    """
    cleaned_text = []
    regex = re.compile(r'[<>]')
    for line in open(filename, "r"):
        if not regex.search(line) and not line.startswith("STEP") and len(line) > 10:
            cleaned_text.append(line)
    return cleaned_text

def cleanFiles(filenames: list, pb: bool = False) -> list:
    """
    Apply cleanFile on a list of files.

    Parameters
    ----------
    filenames : list
        List of dataset directories.
    pb : bool, optional
        Show progress bar of finished files.

    Returns
    -------
    list
        List of list of strings as cleaned dataset lines.

    """
    all_cleaned_text = []
    for i, filename in enumerate(filenames):
        cleaned_text = cleanFile(filenames[i])
        all_cleaned_text.append(cleaned_text)
        if pb: print(f"Files done: {i+1}/{len(filenames)}")
    if pb: print("DONE!")
    return all_cleaned_text
        
def dump(data: list, filename: str = "temp") -> None:
    """
    Quick dump some data to a temporary file

    Parameters
    ----------
    data : list
        Data generated from other functions.
    filename : str, optional
        Name of the temporary file. The default value is "temp"

    Returns
    -------
    None.

    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
def dumpRead(filename: str = "temp") -> list:
    """
    Quick read dumped data

    Parameters
    ----------
    filename : str, optional
        Name of the file to read. The default value is "temp"

    Returns
    -------
    list
        Previously saved data by using dump().

    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def head(data: list, n_lines: int = 5) -> None:
    """
    Print the first elements of a given dataset

    Parameters
    ----------
    data : list
        Dataset.
    n_lines : int, optional
        Number of lines to display. The default is 5.

    Returns
    -------
    None
        DESCRIPTION.

    """
    for i, line in enumerate(data):
        if i < n_lines:
            print(f"{i}\t{line}")
            
def formatFile(data: list) -> list:
    """
    Format a pos-tagged database.
    Remove whitespace and punctuation from end and start of words.
    Get all words lowercase

    Parameters
    ----------
    data : list
        Database to format.

    Returns
    -------
    list
        Formatted dataset.

    """
    cleaned_db = []
    for sentence in data:
        cleaned_sentence = []
        for word, tag in sentence:
            cleaned_sentence.append((word.strip(string.punctuation + " ").lower(), tag))
        cleaned_db.append(cleaned_sentence)
    return cleaned_db

def read_csv(csv: str = 'data/nyt-ingredients-snapshot-2015.csv') -> list:
    """
    Read a given CSV file and save rows as separate dictionaries

    Parameters
    ----------
    data : str
        Directory to csv file

    Returns
    -------
    list
        List of rows as dictionaries

    """
    data = pd.read_csv(csv)
    keys = data.columns
    values = data.values
    rows = []
    for value in values:
        d = {}
        for i, key in enumerate(keys):
            d[key] = value[i]
        rows.append(d)
    return rows
    
def remove_plurals(low: list) -> list:
    return list(set([singular(word) for word in low]))
    
    
    
    
    