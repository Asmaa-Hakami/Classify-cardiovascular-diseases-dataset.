import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns


#Load the data from CSV file.
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")


#split the dataset into training and test datasets
training, testing  = train_test_split(data,test_size=0.30)

Y_test = testing.iloc[:, 12].values
X_test = testing.iloc[:, :-1]

#Split training data by class
ytrain_death = training[training ['DEATH_EVENT'] == 1]
ytrain_non_death = training[training ['DEATH_EVENT'] == 0]


#Mean vector and variance of all features in each class
means_death = ytrain_death.groupby('DEATH_EVENT').mean()
means_non_death = ytrain_non_death.groupby('DEATH_EVENT').mean()
variance_death = ytrain_death.groupby('DEATH_EVENT').var()
variance_non_death = ytrain_non_death.groupby('DEATH_EVENT').var()

print("Mean of each feature in class 1 (death)\n",means_death)
print("*****************************************************\n")
print("Mean of each feature in class 0 (non death)\n",means_non_death)
print("*****************************************************\n")
print("Variance of each feature in class 1 (death)\n",variance_death)
print("*****************************************************\n")
print("Variance of each feature in class 0 (non death)\n",variance_non_death)
print("*****************************************************\n")

# Total rows
total_ppl = training.count().values[0]
# Prior probability of class death 
p_death = ytrain_death['DEATH_EVENT'].count()/total_ppl
# Prior probability of class non death 
p_non_death = ytrain_non_death['DEATH_EVENT'].count()/total_ppl

print("\nPrior probability of class 1 is ",p_death)
print("Prior probability of class 0 is ",p_non_death)


#Gaussian probability method
def p_x_given_y(x, mean_y, variance_y):

    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    
    return p


#-----------------------------Testing-----------------------------

y_predect = [] #store prodection of class of each row
X_test = X_test.to_numpy()

for i in range (0, X_test.shape[0]):
    p_test_death = p_death *\
    p_x_given_y(X_test[i][0], means_death['age'].values[0],variance_death['age'].values[0])*\
    p_x_given_y(X_test[i][1], means_death['anaemia'].values[0], variance_death['anaemia'].values[0])*\
    p_x_given_y(X_test[i][2], means_death['creatinine_phosphokinase'].values[0], variance_death['creatinine_phosphokinase'].values[0])*\
    p_x_given_y(X_test[i][3], means_death['diabetes'].values[0], variance_death['diabetes'].values[0])*\
    p_x_given_y(X_test[i][4], means_death['ejection_fraction'].values[0], variance_death['ejection_fraction'].values[0])*\
    p_x_given_y(X_test[i][5], means_death['high_blood_pressure'].values[0], variance_death['high_blood_pressure'].values[0])*\
    p_x_given_y(X_test[i][6], means_death['platelets'].values[0], variance_death['platelets'].values[0])*\
    p_x_given_y(X_test[i][7], means_death['serum_creatinine'].values[0], variance_death['serum_creatinine'].values[0])*\
    p_x_given_y(X_test[i][8], means_death['serum_sodium'].values[0], variance_death['serum_sodium'].values[0])*\
    p_x_given_y(X_test[i][9], means_death['sex'].values[0], variance_death['sex'].values[0])*\
    p_x_given_y(X_test[i][10], means_death['smoking'].values[0], variance_death['smoking'].values[0])*\
    p_x_given_y(X_test[i][11], means_death['time'].values[0], variance_death['time'].values[0])

    p_test_non_death = p_non_death *\
    p_x_given_y(X_test[i][0], means_non_death['age'].values[0],variance_non_death['age'].values[0])*\
    p_x_given_y(X_test[i][1], means_non_death['anaemia'].values[0], variance_non_death['anaemia'].values[0])*\
    p_x_given_y(X_test[i][2], means_non_death['creatinine_phosphokinase'].values[0], variance_non_death['creatinine_phosphokinase'].values[0])*\
    p_x_given_y(X_test[i][3], means_non_death['diabetes'].values[0], variance_non_death['diabetes'].values[0])*\
    p_x_given_y(X_test[i][4], means_non_death['ejection_fraction'].values[0], variance_non_death['ejection_fraction'].values[0])*\
    p_x_given_y(X_test[i][5], means_non_death['high_blood_pressure'].values[0], variance_non_death['high_blood_pressure'].values[0])*\
    p_x_given_y(X_test[i][6], means_non_death['platelets'].values[0], variance_non_death['platelets'].values[0])*\
    p_x_given_y(X_test[i][7], means_non_death['serum_creatinine'].values[0], variance_non_death['serum_creatinine'].values[0])*\
    p_x_given_y(X_test[i][8], means_non_death['serum_sodium'].values[0], variance_non_death['serum_sodium'].values[0])*\
    p_x_given_y(X_test[i][9], means_non_death['sex'].values[0], variance_non_death['sex'].values[0])*\
    p_x_given_y(X_test[i][10], means_non_death['smoking'].values[0], variance_non_death['smoking'].values[0])*\
    p_x_given_y(X_test[i][11], means_non_death['time'].values[0], variance_non_death['time'].values[0])

    #print(p_test_death)
    #print(p_test_non_death)
    #print("\n")
    correct_p = max(p_test_death, p_test_non_death)
    if correct_p == p_test_non_death:
        y_predect.append(0)
    else:
        y_predect.append(1)

print("\nThe prediction array:\n",y_predect)

#Print accuracy, precision, and recall.
print("\nAccuracy: ", accuracy_score(Y_test, y_predect))
print(classification_report(Y_test, y_predect))

#Plot a pair plot of the data
sns.pairplot(data, hue = 'DEATH_EVENT')
plt.show()

