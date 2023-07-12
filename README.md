# Census income prediction
Project that build a machine learning model to predict 2 different income level. Multiple model are being used along with GridSearch for hyperparameter tuning.
Decent result are produced and compared along with visuals using Seaborn & matplotlib libraries. </br>
Code can be found on [census_income_prediction.ipynb](https://github.com/ricocahyadi777/census-income-prediction/blob/main/census_income_prediction.ipynb)

## Overview
Extraction was done by Barry Becker from the 1994 Census database.  A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.

Dataset and more info can be found on https://archive.ics.uci.edu/dataset/20/census+income

Listing of attributes:

- Y variable is >50K, <=50K.

- age: continuous. 
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

## Preprocess

Removing all the missing values
```python
# Remove the missing value labelled with "?"
census_train = census_train[(census_train.astype(str) != '?').all(axis=1)]
census_test = census_test[(census_test.astype(str) != '?').all(axis=1)]
```

Removing native-country column which is not really useful and has a high cardinality
```python
# Dropping the 'native-country' column
census_train = census_train.drop(['native-country'], axis=1)
census_test = census_test.drop(['native-country'], axis=1)
```

Defining all the categorical variable so that it can be encoded later
```python
# Define the category column
category_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
```

Get the y variable, and then change it to binary format
```python
# y variable changed into binary format and assign it
y_train = census_train.apply(lambda row: 1 if '>50K'in row['income'] else 0, axis=1)
y_test = census_test.apply(lambda row: 1 if '>50K'in row['income'] else 0, axis=1)
# Then remove the income column
census_train = census_train.drop(['income'], axis=1)
census_test = census_test.drop(['income'], axis=1)
```

Encoding all the categorical variable using Integer Encoding
```python
# Encode the category into numerical form
# Set the x variable for the Encoding
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for column in category_columns:
    label_encoder = enc.fit(census_train[column])
    print("Categorical classes:", label_encoder.classes_)
    integer_classes = label_encoder.transform(label_encoder.classes_)
    print("Integer classes:", integer_classes)
    train = label_encoder.transform(census_train[column])
    test = label_encoder.transform(census_test[column])
    census_train[column] = train
    census_test[column] = test
```
Selection of x_train and x_test depend on how you encode the categorical variable.
For my code i am using the Integer encoding since the dimensionality will be high if i use the dummy.
Especially when the result does not differ that much

If you wanted to use One-Hot Encoding instead
```python
# Encode the category using one hot encoder
census_train_dummies = pd.get_dummies(census_train, columns=category_columns)
census_test_dummies = pd.get_dummies(census_test, columns=category_columns)
```

Data after the preprocessing step
![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/a755c34f-1ac2-496a-905b-bdaac6969719)

## Model Building
Building a default Decision Tree model
```python
from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=2019)
clf.fit(x_train, y_train)
```

Let's check the result for the model </br>
![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/5a461e6e-72c6-4090-8231-88259cb02ee1)

The test data shows a good result, but how about we check the train data? </br>
![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/3a6860ce-0457-457b-ba6e-3bf7d598b939)

Almost perfectt!!! </br>
We can see that all the metrics for train data are almost perfect, while test data only have 80% accuracy with other metrics performing worse.
This clearly display that there are overfitting

Now let's try checking the ROC curve.
```python
import matplotlib.pyplot as plt
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[0], tpr[0], _ = metrics.roc_curve(y_test, y_pred)
fpr[1], tpr[1], _ = metrics.roc_curve(y_train, y_pred_train)
for i in range(2):
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
plt.figure(figsize = (5, 5))    
plt.plot([0, 1], [0, 1], linestyle='--')

colors = (['aqua', 'crimson'])
data_set = (['test', 'train'])
for i, data, color in zip(range(2), data_set, colors):
    plt.plot(fpr[i], tpr[i], color=colors[i],
             label='ROC curve of {0} data (Area = {1:0.2f})'
             ''.format(data, roc_auc[i]))    

plt.xlabel('False positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
```

![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/6d3905a3-5a0c-496d-abcd-91ad73961d62)

We can see from the ROC curve too, overfitting clearly happened since the curve of test and training are so far away from one another.

## Modified Model
Now let's build a modified model of the decision tree.
One of the most important thing in decision tree is pruning the tree. We can do so by configuring the 'max_depth' parameter. Now let's iterate through multiple 'max-depth' parameter.
```python
# To get the best max depth
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
max_depth = np.arange(1, 20, 1)
train_results = []
test_results = []

for i in max_depth:
    dt_estimator = tree.DecisionTreeClassifier(max_depth=i, min_samples_split=150, random_state=2019, criterion='entropy')
    dt_estimator.fit(x_train, y_train)
    train_pred = dt_estimator.predict(x_train)
    accuracy_train = accuracy_score(y_train, train_pred)
    train_results.append(accuracy_train)
    test_pred = dt_estimator.predict(x_test)
    accuracy_test = accuracy_score(y_test, test_pred)
    test_results.append(accuracy_test)
    
plt.plot(max_depth, train_results, label="Training accuracy score", color="red")
plt.plot(max_depth, test_results, label="Test accuracy score", color="blue")
plt.title("Accuracy based on tree depth")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.legend(loc="upper left")
plt.show()
```

![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/0d1c2b28-a46b-4949-99b6-047f95232acd)
From this image we can see the depth of '10' is quite optimal, it gives good accuracy, but the test and train accuracy is not that far apart.
Thus, let's build a decision tree with these parameters.
```pyhton
clf_modified = tree.DecisionTreeClassifier(class_weight='None', criterion='entropy', max_depth=10,
                                            max_features=None, max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            min_samples_leaf=1, min_samples_split=150,
                                            min_weight_fraction_leaf=0.0,  random_state=2019,
                                            splitter='best')
clf_modified.fit(x_train, y_train)
```
Result: </br>
![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/d9c3ad66-816a-4386-98b8-f60a32e8cd0f) ![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/e4178903-4558-46b4-973c-da000e93a679)

![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/20718c84-0f29-4c01-bc12-d1f0b302fc23)

## Comparison
We can see the result are much better. Tuning the parameter will improve the decision tree because we prune the tree to avoid overfitting.
From my experiment, my modified parameter are (Excluding random state):
- criterion='entropy'
- max_depth=10
- min_samples_split=150

I choose this parameter because i want to find balance, when i set the class weight into 'balanced', the ROC are greatly improved however the accuracy is barely improve. So I want to get decent improvement on both side.

![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/46cbb0cf-5bde-447b-b3c7-8586e44a2c76)

## Ensemble Learning

In this machine learning task, we should interpret precision as <b>precentage of correctly predicted those with income above 50k out of all those labelled with income above 50k</b>, and we should interpret recall as <b>precentage of correctly predicted those with income above 50k out of all who really have income above 50k.</b>

My choice of the better metric between $F_2$ and $F_{0.5}$ is $F_2$. <b>Because we want to classify correctly as many positive samples as possible, rather than maximizing the number of correct classifications. This is done with the assumption that we are a business, and wanted to market our product to public, but our product is a bit expensive. So, those with income higher than 50k will be more likely to purchase. Subsequently, we want to get as many target we can market our product to. Thus we are focusing more on recall. </b>

### Random Forest
Objective metric on test:  𝐹2
 
Parameters: criterion, max_depth, min_samples_split, n_estimators
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, fbeta_score

clfRF=RandomForestClassifier(random_state=2019)
parameter_space_rf = { 
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [100, 200],
    'max_depth': [5, 10, 15]
}
f_scorer = make_scorer(fbeta_score, beta = 2)
CV_rfc = GridSearchCV(estimator=clfRF, param_grid=parameter_space_rf, scoring = f_scorer, cv= 5)
CV_rfc.fit(x, y)
CV_rfc.best_params_
```
Best result:
- 'criterion': 'gini'
- 'max_depth': 15
- 'min_samples_split': 100
- 'n_estimators': 100

Now we just need to fit the data with those parameters.
```python
# Fitting the data with the best parameters
clfRF=RandomForestClassifier(random_state=2019,**CV_rfc.best_params_)
clfRF.fit(x_train,y_train)
```
<b>Result:</b>

![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/817e0c0d-30b4-4ebd-809b-c414c24a4ade)

<b>Important Feature (With Visualization):</b>

```python
feature_imp_rf = pd.Series(clfRF.feature_importances_,index=x_train.columns).sort_values(ascending=False)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp_rf, y=feature_imp_rf.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()
```
![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/2b933a7a-0d3b-4bf2-8b68-0a1914c784c3)

### AdaBoost
Objective metric on test:  𝐹2
 
Parameters: algorithm, learning_rate, n_estimators
```python
from sklearn.ensemble import AdaBoostClassifier

clfAB=AdaBoostClassifier(random_state=2019)

parameter_space_ab = { 
    'n_estimators': [50, 100],
    'algorithm': ['SAMME', 'SAMME.R'],
    'learning_rate': [0.4, 0.5, 0.6]
}
f_scorer = make_scorer(fbeta_score, beta = 2)
CV_ab = GridSearchCV(estimator=clfAB, param_grid=parameter_space_ab, scoring = f_scorer, cv= 5)
CV_ab.fit(x, y)
CV_ab.best_params_
```

Best result:
- 'algorithm': 'SAMME.R'
- 'learning_rate': 0.6
- 'n_estimators': 100

Now we just need to fit the data with those parameters.
```python
clfAB=AdaBoostClassifier(random_state=2019, **CV_ab.best_params_)
clfAB.fit(x_train,y_train)
```
<b>Result:</b>

![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/5d0f2423-72bc-4fe4-9dbf-b8b557703f57)

<b>Important Feature (With Visualization):</b>
```python
feature_imp_ab = pd.Series(clfAB.feature_importances_,index=x_train.columns).sort_values(ascending=False)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp_ab, y=feature_imp_ab.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()
```

![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/34c36cd5-9f26-4069-a695-a0ba6abe1255)

### Gradient Boosting
Objective metric on test:  𝐹2
 
Parameters: criterion, max_depth, min_samples_split, n_estimators
```python
from sklearn.ensemble import GradientBoostingClassifier

clfGB=GradientBoostingClassifier()
parameter_space_gb = { 
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [100, 200],
    'criterion': ['squared_error', 'friedman_mse']
}
f_scorer = make_scorer(fbeta_score, beta = 2)
CV_gb = GridSearchCV(estimator=clfGB, param_grid=parameter_space_gb, scoring = f_scorer, cv= 5)
CV_gb.fit(x, y)
CV_gb.best_params_
```

Best result:
- 'criterion': 'squared_error'
- 'max_depth': 10
- 'min_samples_split': 200
- 'n_estimators': 200

Now we just need to fit the data with those parameters.
```python
clfGB=GradientBoostingClassifier(random_state=2019, **CV_gb.best_params_)
clfGB.fit(x_train,y_train)
```

<b>Result:</b>

![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/cf0da09a-a428-4c94-a414-c332c185a818)

<b>Important Feature (With Visualization):</b>
```python
feature_imp_gb = pd.Series(clfGB.feature_importances_,index=x_train.columns).sort_values(ascending=False)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Creating a bar plot
sns.barplot(x=feature_imp_gb, y=feature_imp_gb.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()
```

![image](https://github.com/ricocahyadi777/census-income-prediction/assets/63791918/83ed5fed-2e3b-4251-9128-3ca607195654)

### Final Comparison of IMportant Features between all models
| Classifier | RandomForest | AdaBoost | GradientBoosting |
| --- | --- | --- | --- |
| Feature 1 |relationship|capital-gain|relationship|
| Feature 2 |capital-gain|occupation|capital-gain|
| Feature 3 |education-num|capital-loss|education-num|
| Feature 4 |marital-status|age|fnlwgt|
| Feature 5 |age|relationship|age|
