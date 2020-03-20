import pandas as pd
#import matplotlib.pyplot as plt
fraud_data = pd.read_csv("C:/Training/Analytics/Decison_Tree/Fraud_check/Fraud_check.csv")
fraud_data_ori = fraud_data
fraud_data.head()

fraud_data.describe()

import numpy as np


#dummy variables
fraud_dummies = pd.get_dummies(fraud_data[["Undergrad","Marital.Status","Urban"]])
fraud_dummies = fraud_dummies.drop(['Undergrad_NO','Marital.Status_Divorced','Urban_NO'],axis=1)

fraud_data = pd.concat([fraud_data,fraud_dummies],axis=1)
fraud_data=fraud_data.drop(['Undergrad','Marital.Status','Urban','Risk_Factor'],axis=1)


fraud_data['Risk_Factor'] = pd.cut(x=fraud_data['Taxable.Income'], bins=[1, 30000,100000], labels=['Good', 'Risky'], right=False)
fraud_data['Risk_Factor'].unique()
fraud_data.Risk_Factor.value_counts()

colnames = list(fraud_data.columns)
predictors = colnames[1:7]
target = colnames[7]


# Splitting fraud_data into training and testing fraud_data set


# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
#fraud_data['is_train'] = np.random.uniform(0, 1, len(fraud_data))<= 0.75
#fraud_data['is_train']
#train,test = fraud_data[fraud_data['is_train'] == True],fraud_data[fraud_data['is_train']==False]

from sklearn.model_selection import train_test_split
train,test = train_test_split(fraud_data,test_size = 0.3)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
model1 = model.fit(train[predictors],train[target])
#Training fraud_data


train_preds = model.predict(train[predictors])
pd.Series(train_preds).value_counts()
pd.crosstab(train[target],train_preds)

# Accuracy = train
np.mean(train.Risk_Factor == model.predict(train[predictors]))

#Testing fraud_data

preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = Test
np.mean(preds==test.Risk_Factor) 


from sklearn import tree
#tree.plot_tree(model1.fit(iris.fraud_data, iris.target)) 

clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(train[predictors], train[target])
tree.plot_tree(clf) 

#cluster_labels=preds
#iris_train = train
#iris_train['clust']=cluster_labels # creating a  new column and assigning it to new column 
#iris_train = iris_train.iloc[:,[5,0,1,2,3,4]]
#iris_train.head()
#
#cluster_labels=preds
#iris_test = test
#iris_test['Predicted Species']=cluster_labels # creating a  new column and assigning it to new column 
##iris_test = iris_test.iloc[:,[5,0,1,2,3,4]]
#iris_test.head()

