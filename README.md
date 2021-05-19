# NLTK RNN recipe generator

## Introduction
To combat food waste and inspire people with recipe ideas we thought of creating a recipe generator. The algorithm will train on a database of existing food recipes. The user will input the ingredients that he/she has left in their fridge. Accordingly, the recipe generator will generate a recipe with these ingredients. Alternatively, for those who hit a creative burnout with their cooking, supplying no ingredients to the model will result in a random selection of ingredients. The LSTM RNN model is trained on a recipe database, and accordingly learn combinations and portions of ingredients that work well together. 

 
## Research question
Can we create a recipe generator model that generates recipes based on given ingredients or random selecetion of ingredients?


## The repository contains the following:

#### 1. FinalCode.ipynb
A jupyter notebook that conains all code used in the project. 

#### 2. Final report
Our final project report 

#### 3. Presentation
Slide deck for class presentation

#### 4. Data folder
Contains all datasets that been used and generated.
- ArmedForcesRecipes.exl - a dataset of armed forces recipes taken from https://esha.com/resources/additional-databases/.
- CommonRecipes.exl - a dataset of common recipes taken from https://esha.com/resources/additional-databases/.
- VegetarianRecipes.exl - a dataset of vegeterian recipes taken from https://esha.com/resources/additional-databases/.
- train.json - a training set containing recipes id, type of cuisine, and list of ingredients taken from https://www.kaggle.com/kaggle/recipe-ingredients-dataset.
- test.json - a test set containing recipes id and list of ingredients taken from https://www.kaggle.com/kaggle/recipe-ingredients-dataset.
- final_ingr.csv - a list of 7137 ingredients extracted from train.json and test.json files.
- ingredients.csv - a list of 10000 ingredients extracted from https://world.openfoodfacts.org/ingredients.json used for pos-tagging of ingredients
- nyt-ingredients-snapshot-2015.csv - extracted recipes sentences taken from https://github.com/nytimes/ingredient-phrase-tagger.
- recipes_armed.csv - armed forces recipes after preprocessing
- recipes_veg.csv - vegeterian recipes after preprocessing
- recipes_common.csv - common recipes after preprocessing
- Cococcs of ingredients - co-occurrences of ingredients 
- Cococcs of ingredients vs verbs - co-occurrences of ingredients vs verbs
- final_shortened - processed data file used to train the LSTM model
- inglist_final - final, processed ingredients list with no duplicates and no plural words
- weights-20-3.1225.hdf5 - the model we used for text generation, to avoid training the model over and pver every time

#### 5. Code Folder
Contains all jupyter notebooks and spyder scripts we worked on throughout the project. All the code presented in this folder can be found in 'final code' notebook mentioned above.
- helpers.py - helper functions to ease the cleaning and formatting processes of the data.
- preprocess.py - lemmatization, pos-tagging and ingredients extraction 
- Preprocessing - Separating Recipes.ipynb - apply preprocessing on recipes datasets
- Correlation of ingredients.ipynb - co-occurrences of ingredients vs ingredients and ingredients vs verbs
- lstm.py - our LSTM model 
