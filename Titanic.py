#import libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,accuracy_score
import warnings
warnings.filterwarnings('ignore')

#Load train data 
train_data = pd.read_csv("F:\\project\\Kaggel\\Titanic\\train.csv")
train_data.head()

#load test data 
test_data = pd.read_csv("F:\\project\\Kaggel\\Titanic\\test.csv")
test_data.head()


#load gender submission
submission = pd.read_csv("F:\\project\\Kaggel\\Titanic\\gender_submission.csv")
print(submission.shape)


submission =pd.DataFrame(submission)
submission.drop("PassengerId",axis=1,inplace=True)
print(submission)


train_data.isnull().sum()
test_data.isnull().sum()

#plot correlation between data 
fig ,ax = plt.subplots(figsize=(12,8))
cor = train_data.select_dtypes(include = [np.number])
sns.heatmap(cor.corr(),annot = True,cmap = "YlGnBu",fmt=".2f")

#drow relation between fare and survived 
plt.scatter(x = train_data["Fare"], y =train_data["Survived"])
plt.xlabel("Fare")
plt.ylabel("Survivde")
plt.title("Fare vs Survived")
plt.show()


#Delete anomaly value 
train_data = train_data.drop(train_data[(train_data["Fare"]>400)].index)

#drow relation between fare and survived 
plt.scatter(x = train_data["Fare"], y =train_data["Survived"])
plt.xlabel("Fare")
plt.ylabel("Survivde")
plt.title("Fare vs Survived")
plt.show()


print(train_data["Age"].describe())
sns.histplot(train_data["Age"])
plt.show()

#impute train data "Age"
impute = SimpleImputer(missing_values = np.nan, strategy= "mean",)
train_data["Age"] = impute.fit_transform(train_data[["Age"]])

print(train_data["Age"].describe())
sns.histplot(train_data["Age"])
plt.show()

#impute test_data "Age"
impute = SimpleImputer(missing_values = np.nan, strategy= "mean")
test_data["Age"] = impute.fit_transform(test_data[["Age"]])


train_data.drop(columns=["Cabin","Name","PassengerId","Ticket"],inplace=True,axis=1)
test_data.drop(columns=["Cabin","Name","PassengerId","Ticket"],inplace=True,axis=1)


train_data.dropna(inplace=True,axis=0)
#impute test_data "Age"
impute = SimpleImputer(missing_values = np.nan, strategy= "mean")
test_data["Fare"] = impute.fit_transform(test_data[["Fare"]])



#drow number of male and female were survive or not 
train_data = train_data.reset_index(drop=True)
survived_male = 0
non_survived_male = 0
survived_female = 0
non_survived_female =0
for i in range(len(train_data["Sex"])):
    if train_data.loc[i,"Sex"]== "male" and train_data.loc[i,"Survived"]==1:
        survived_male += 1 
    elif train_data.loc[i,"Sex"]== "male" and train_data.loc[i,"Survived"]==0:
        non_survived_male += 1 
    elif train_data.loc[i,"Sex"]=="female" and train_data.loc[i,"Survived"]==1:
        survived_female +=  1
    else :
        non_survived_female += 1
print(f"Number of survived males: {survived_male}")
print(f"Number of non-survived males: {non_survived_male}")
print(f"Number of survived females: {survived_female}")
print(f"Number of non-survived females: {non_survived_female}")
sns.histplot(train_data["Sex"])
plt.title("Sex : Male and Female")

#count number of survived male and female
data = {
    "Category": ["Survived Male", "Non-Survived Male", "Survived Female", "Non-Survived Female"],
    "Count": [survived_male, non_survived_male, survived_female, non_survived_female]
}

# Convert to DataFrame
count_data = pd.DataFrame(data)

# Plot the counts
sns.barplot(data=count_data,x="Category", y="Count")
plt.title("Survival Counts by Sex")
plt.xticks(rotation=90)
plt.show()

numeric_train = train_data.select_dtypes(include=[np.number]).columns.tolist()
object_train = train_data.select_dtypes(exclude=[np.number]).columns.tolist()
train_data[object_train]

numeric_test = test_data.select_dtypes(include=[np.number]).columns.tolist()
object_test = test_data.select_dtypes(exclude = [np.number]).columns.tolist()

dummy_train = pd.get_dummies(train_data[object_train],drop_first=True)


dummy_test = pd.get_dummies(test_data[object_test],drop_first=True)


train_data.drop(columns= object_train,inplace=True)
train_data = pd.concat([train_data,dummy_train],axis=1)


test_data.drop(columns=object_test,inplace=True)
test_data = pd.concat([test_data,dummy_test],axis=1)

#Splitting data 
X_train = train_data.iloc[:,1:]
y_train = train_data.iloc[:,0]
X_test = test_data
y_test = submission.values.ravel()
print("shape of X_train is :",X_train.shape)
print("shape of y_train is :",y_train.shape)
print("shape of X_test is :",X_test.shape)
print("shape of y_test is :",y_test.shape)

#Apply Model 
LogisticRegressionModel = LogisticRegression(max_iter=200,n_jobs=-1,random_state=44)
LogisticRegressionModel.fit(X_train,y_train)
print("score of train is :",LogisticRegressionModel.score(X_train,y_train))
print("Score of test is :",LogisticRegressionModel.score(X_test,y_test))


#predicted value
y_pred_LR = LogisticRegressionModel.predict(X_test)
y_pred_prob =LogisticRegressionModel.predict_proba(X_test)
print("the predicted value is :",y_pred_LR[:10])
print("the actual value is :   ",y_test[:10])
print("the probability of X_test is :\n",y_pred_prob[:5])

print("the accurecy score for Logistic Regression is :",accuracy_score(y_test,y_pred_LR))
print("f1_score for Logistic Regression is :",f1_score(y_test,y_pred_LR))
print("number of iteration is :",LogisticRegressionModel.n_iter_)

CM = confusion_matrix(y_test,y_pred_LR)
sns.heatmap(CM,fmt="d",cmap="Blues",annot=True)
plt.title("confusion_matrix")
plt.xlabel("Actual value")
plt.ylabel("Predicted value")
plt.show()
print("features are: ",X_train.columns)
print("for test:",train_data.columns)
import joblib 
joblib.dump(LogisticRegressionModel,"LRModel.pkl")
print("the model saved successfully")