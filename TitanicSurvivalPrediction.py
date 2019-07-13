#Python program to predict Titanic survival with machine learning
#Author - Hitesh Thakur

#Importing required libraries
import pandas as pd;
import numpy as np;
from copy import deepcopy;
import seaborn as sns;
import matplotlib.pyplot as plt;
from sklearn.preprocessing import LabelEncoder;
from sklearn.svm import SVC;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import confusion_matrix,accuracy_score;
from sklearn.ensemble import RandomForestClassifier;

##DATA DESCRIPTION
# Variable	Definition  	Key
# survival	Survival	    0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	    Sex
# Age	    Age in years
# sibsp	    # of siblings / spouses aboard the Titanic
# parch	    # of parents / children aboard the Titanic
# ticket	Ticket number
# fare	    Passenger fare
# cabin	    Cabin number
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

#Reading data from .csv file using pandas
titanic_data=pd.read_csv('train.csv');

#Checking presence of null values
print(titanic_data.isnull().sum());

# Replace NaN with Not Available
titanic_data.fillna({'Age' : titanic_data['Age'].mean().round(2) ,'Cabin' : 'Cabin-Not-Available','Embarked' : 'Embarked-Not-Available'},inplace=True);

#Replace 1,2,3 with firstClass,secondClass,thirdClass
titanic_data['Pclass'].replace({1:'firstClass',2:'secondClass',3:'thirdClass'},inplace=True);

#Label encoding : Sex
LE=LabelEncoder();
titanic_data['Sex']=LE.fit_transform(titanic_data['Sex']);

#Dropping irrelevant features
titanic_data.drop(columns=['PassengerId','Name','Ticket'],inplace=True);

#One hot Encoder : Pclass,Cabin,Embarked
Titanic=pd.DataFrame();
for x in ['Pclass','Cabin','Embarked']:
    dummy_variable=pd.get_dummies(titanic_data[x],sparse=True);
    dummy_variable.drop(dummy_variable.columns[0],axis='columns',inplace=True);
    Titanic = pd.concat([Titanic, dummy_variable], axis='columns');

Titanic=pd.concat([Titanic,titanic_data['Sex'],titanic_data['Age'],titanic_data['SibSp'],titanic_data['Parch'],titanic_data['Fare']],axis='columns');
Y=np.array(titanic_data['Survived']);

#Splitting data into two sets: Train and Test
X_train,X_test,Y_train,Y_test=train_test_split(Titanic,Y,test_size=0.3,random_state=0);

#Training model with preprocessed dataset
random_forest = RandomForestClassifier(1000);
random_forest.fit(X_train, Y_train);
Y_pred = random_forest.predict(X_test);
random_forest_accuracy=accuracy_score(Y_test,Y_pred);

#Accuracy of the model
print(random_forest_accuracy);