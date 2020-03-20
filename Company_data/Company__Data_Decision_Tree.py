import pandas as pd
#import matplotlib.pyplot as plt
company_data = pd.read_csv("C:/Training/Analytics/Decison_Tree/Company_data/Company_data.csv")
company_data_ori = company_data
company_data.head()

decriptive=company_data.describe()

import numpy as np

# =============================================================================
# Data Manipulation
# =============================================================================

#dummy variables
company_dummies = pd.get_dummies(company_data[["ShelveLoc","Urban","US"]])

company_data = pd.concat([company_data,company_dummies],axis=1)
company_data = company_data.drop(['ShelveLoc','Urban','US','ShelveLoc_Bad','Urban_No','US_No'],axis=1)


def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(company_data.iloc[:,0:])

# =============================================================================
# 
# =============================================================================




company_data['Sales_Category'] = pd.cut(x=company_data['Sales'], bins=[0,5,9,20], labels=['Low', 'Medium','High'], right=False)
company_data['Sales_Category'].unique()
company_data.Sales_Category.value_counts()

colnames = list(company_data.columns)
predictors = colnames[1:12]
target = colnames[12]

# Splitting company_data into training and testing company_data set


# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
#company_data['is_train'] = np.random.uniform(0, 1, len(company_data))<= 0.75
#company_data['is_train']
#train,test = company_data[company_data['is_train'] == True],company_data[company_data['is_train']==False]

from sklearn.model_selection import train_test_split
train,test = train_test_split(company_data,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
model1 = model.fit(train[predictors],train[target])
#Training company_data


train_preds = model.predict(train[predictors])
pd.Series(train_preds).value_counts()
pd.crosstab(train[target],train_preds)

# Accuracy = train
np.mean(train.Sales_Category == model.predict(train[predictors]))

#Testing company_data

preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = Test
np.mean(preds==test.Sales_Category) 


# =============================================================================
# from sklearn import tree
# #tree.plot_tree(model1.fit(iris.company_data, iris.target)) 
# 
# clf = tree.DecisionTreeClassifier(random_state=0)
# clf = clf.fit(train[predictors], train[target])
# tree.plot_tree(clf) 
# 
# =============================================================================
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

