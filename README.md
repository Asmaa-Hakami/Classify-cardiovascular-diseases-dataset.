# Classify-cardiovascular-diseases-dataset.
The cardiovascular diseases dataset was classified using Naïve Bayes from scratch and KNN built-in function Algorithms.

Naïve Bayes Classifiers steps:
1. Load the data from CSV file.
2. Shuffle and split the dataset into training and test datasets (use 30/70 ratio)
3. Split your training data by class, compute the mean vector and variance of the feature
in each class.
4. Calculate the prior probabilities and Gaussian probability distribution for each
features and class.
5. Make prediction on the testing set.
6. Compute the accuracy, precision, and recall.
7. Plot a scatter pair plot of the data, coloring each data point by its class.

KNN Classifier steps:
1. Load the dataset from CSV file.
2. Divide the dataset into training and testing.
3. Train the KNN algorithm.
4. Test the algorithm on the test dataset with different values of k=1,3,5,10.
5. Report the accuracy, precision, and recall.
6. Report k that gives the highest accuracy
