# Creating an AI recipe generator

## Abstract
To combat food waste and inspire people with recipe ideas we thought of creating a recipe generator. The algorithm will train on a database of existing food recipes. The user will input the ingredients that he/she has left in their fridge. Accordingly, the recipe generator will generate a recipe with these ingredients. Possibly, dietary preferences such as vegetarianism, gluten-free or similar can be incorporated. The model will be trained on a recipe database, and accordingly learn combinations and portions of ingredients that work well together. 

 
## Research questions
- What recipe can we make with the ingredients we have in my fridge?
- What ingredients appear most often? And is it different between cuisines?
- What ingredients co-appear most often? And which do not correlate? 
- For a given dish, which dishes are most similar?

## Dataset and Setup
1. Clone this repository
2. Download the following files from https://esha.com/resources/additional-databases/
- ArmedForcesRecipes.exl
- CommonRecipes.exl
- VegetarianRecipes.exl
3. Create a folder called "data" in this directory.
4. Create a file named .gitignore in this directory and type "data/" inside.
5. Move the downloaded datasets into the data folder.


## A tentative list of milestones for the project
Tasks distribution will be determined later, with fair work distribution in mind.
- Download datasets of recipes and merge them to create a one big united dataset (ideally, if we find more than one database)
- Exploring the datasets (columns, features, shape etc)
- Preprocessing (choose features. e.g. ingredients, cooking time, portions etc.)
- Separate NLP processing for ingredients and instructions
- NLP - for ingredients (quantity, unit, type, food, preparation)
- NLP - for instructions (consists of time, temperature, actions, tools and ingredients)
- Analyze which verbs go with what ingredient (e.g. you don’t fry milk)
- Using LSTM for text generation
- Testing the output recipe (i.e. cook it!)
- Making adjustments if needed 


## Documentation
SOURCES
https://www.kdnuggets.com/2020/07/generating-cooking-recipes-using-tensorflow.html
https://github.com/derekdjia/AI_Generated_Recipes#21-data-cleaning
https://cosylab.iiitd.edu.in/recipedb/ 
