
# coding: utf-8

#########################################################################
########## JEST - Junior Enterprise for Science and Technology ##########
#########################################################################
#########################################################################
#################### Machine Learning Workshop ##########################
#########################################################################


from sklearn import svm
from sklearn import ensemble
import sklearn as sk
import numpy as np
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
get_ipython().magic('matplotlib notebook')

# Titanic Analysis
# Load files
titanicData = pd.read_csv('../data/titanic/train.csv', sep = ',')
list(titanicData)


# Decide which variables to use
# Examples of some differences that can be related to the Survival rate

survived =  titanicData.loc[(titanicData.Survived == 1)] 
perished = titanicData.loc[(titanicData.Survived == 0)]

# Age
plt.figure()
plt.title('Age: Survived vs Perished')
sns.set(color_codes=True)
sns.distplot(survived.Age.loc[~np.isnan(survived.Age)], hist_kws={"alpha": 0.5})
sns.distplot(perished.Age.loc[~np.isnan(perished.Age)], hist_kws={"alpha": 0.5}, color = 'r')
plt.legend(['Survived','Perished'])


# Sex and Class

plt.figure()
plt.title('Survival rate by sex and passenger class')
sns.barplot(x="Sex", y="Survived", hue="Pclass", data = titanicData);


# classifier

titanicData = titanicData.replace(['male', 'female'], [0, 1])
titanicData = titanicData.loc[(~np.isnan(titanicData.Age)) & (~np.isnan(titanicData.Fare))]

indexes = np.random.rand(len(titanicData)) < 0.7
train = titanicData[indexes]
test = titanicData[~indexes]

#classifier1 = svm.SVC()
#classifier1.fit(train[['Pclass', 'Sex', 'Age', 'Fare']], train.Survived)
#predictions = classifier1.predict(test[['Pclass', 'Sex', 'Age', 'Fare']])

classifier1 = svm.SVC()
classifier1.fit(train[['Pclass', 'Sex']], train.Survived)
predictions1 = classifier1.predict(test[['Pclass', 'Sex']])

# Metrics
tn, fp, fn, tp = sk.metrics.confusion_matrix(test.Survived, predictions1).ravel()
accuracy = (tp + tn) / (tp + tn + fn + fp)
sensitivity = tp / (tp + fn)
specificity = tp / (tn + fp)

print('SVM')
print('Accuracy: ', accuracy, '\nSensitivity: ', sensitivity, '\nSpecificity: ', specificity)


# Don't forget that this classifier is not optimized
classifierRF = ensemble.RandomForestClassifier(25)
classifierRF.fit(train[['Pclass', 'Sex', 'Fare', 'Age']], train.Survived)
predictionsRF = classifierRF.predict(test[['Pclass', 'Sex', 'Fare', 'Age']])

tn, fp, fn, tp = sk.metrics.confusion_matrix(test.Survived, predictionsRF).ravel()
accuracy = (tp + tn) / (tp + tn + fn + fp)
sensitivity = tp / (tp + fn)
specificity = tp / (tn + fp)

print('Random Forest')
print('Accuracy: ', accuracy, '\nSensitivity: ', sensitivity, '\nSpecificity: ', specificity)


# https://www.kaggle.com
# https://www.deeplearning.ai

