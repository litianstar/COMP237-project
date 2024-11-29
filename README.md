# COMP237-project
Overview
In this project you will work in groups of five to build a simple text classifier using the  "Bag of words"  language model & the Navie Bayes classifier. As you know movies on Youtube are very common and once a movie is made available many comments are posted over the internet by viewers in response to the movie. But in some instances these comments are auto generated and considered spam.

The purpose of your model is to filter the spam comments by training an Naive Bayes classifier.  The data is available for five movies at the UCI machine learning repository.  

Each team should group will be assigned a movie comments file to work on by the professor.
Requirements
Load the data into a pandas data frame.
Carry out some basic data exploration and present your results. (Note: You only need two columns for this project, make sure you identify them correctly, if any doubts ask your professor)
Using nltk toolkit classes and methods prepare the data for model building, refer to the third lab tutorial in module 11 (Building a Category text predictor ). Use count_vectorizer.fit_transform().
Present highlights of the output (initial features) such as the new shape of the data and any other useful information before proceeding.
Downscale the transformed data using tf-idf and again present highlights of the output (final features) such as the new shape of the data and any other useful information before proceeding.
Use pandas.sample to shuffle the dataset, set frac =1 
Using pandas split your dataset into 75% for training and 25% for testing, make sure to separate the class from the feature(s). (Do not use test_train_ split)
Fit the training data into a Naive Bayes classifier. 
Cross validate the model on the training data using 5-fold and print the mean results of model accuracy.
Test the model on the test data, print the confusion matrix and the accuracy of the model.
