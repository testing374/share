from time import time
t0 = time()
import pandas as pd

##
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Function to plot AUC
def plot_auc(y_test, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

# Function to plot confusion matrix
def conf(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()








# Load the CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\MSNri\OneDrive\桌面\miniProject3\titanic\train.csv")




# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Fill missing values in the 'Age' column with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing values in the 'Fare' column with the median fare
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Drop columns that won't be used for the regression
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)



X = df.drop(columns=['Survived'])
y = df['Survived']

##

from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=8)
X_selected = selector.fit_transform(X,y)

# Create list of all the columns and their score in variables
selected_features = X.columns[selector.get_support()]
feature_scores = selector.scores_[selector.get_support()]

# Create a New Dataframe to store features and their scores
feature_score_df = pd.DataFrame({'Features': selected_features, 'Scores': feature_scores})

# Sort the created dataframe in descending order
feature_score_df = feature_score_df.sort_values(by='Scores', ascending=False)

# Plot a barplot to for better understanding of the features and scores
plt.figure(figsize=(12,8))
ax = sns.barplot(x=feature_score_df['Scores'], y=feature_score_df['Features'])
plt.title('Feature Score', fontsize=18)
plt.xlabel('Scores', fontsize=16)
plt.ylabel('Features', fontsize=16)
for lab in ax.containers:
    ax.bar_label(lab)





##

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# log reg

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'log reg Accuracy: {accuracy}')

# KNN

#Standardizing the features is crucial for KNN and SVM.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_trainS = scaler.fit_transform(X_train)
X_testS = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=5) # You can adjust the number of neighbors
knn.fit(X_trainS, y_train)



y_pred = knn.predict(X_testS)
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Accuracy: {accuracy}')


# SVM

from sklearn.svm import SVC

# Create and train the SVM model
svm = SVC(kernel='linear')  # You can try other kernels like 'rbf' or 'poly'
svm.fit(X_trainS, y_train)



y_pred = svm.predict(X_testS)
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy: {accuracy}')

# DT

from sklearn.tree import DecisionTreeClassifier

# Create and train the Decision Tree model
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)



y_pred = dtree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'DT Accuracy: {accuracy}')


# random forest

from sklearn.ensemble import RandomForestClassifier

# Create and train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'RF Accuracy: {accuracy}')


# naive bayes

from sklearn.naive_bayes import GaussianNB

# Create and train the Naive Bayes model
nb = GaussianNB()
nb.fit(X_trainS, y_train)


y_pred_nb = nb.predict(X_testS)
y_pred_proba_nb = nb.predict_proba(X_testS)[:, 1]

accuracy = accuracy_score(y_test, y_pred_nb)
print(f'NB Accuracy: {accuracy}')

##


import xgboost as xgb

# Create and train the XGBoost model
model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(X_train, y_train)

# Predict probabilities and labels
y_pred_xgb = model_xgb.predict(X_test)
y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_xgb)
print(f'XGB Accuracy: {accuracy}')




##






##
##from sklearn.model_selection import GridSearchCV
##
##
### Define the parameter grid
##param_grid = {
##    'n_estimators': [100, 200, 300],
##    'max_depth': [None, 10, 20, 30],
##    'min_samples_split': [2, 5, 10]
##}
##
### Initialize the Random Forest classifier
##rfCV = RandomForestClassifier(random_state=42)
##
### Initialize GridSearchCV
##grid_search = GridSearchCV(estimator=rfCV, param_grid=param_grid, cv=5, scoring='accuracy')
##
### Fit the model
##grid_search.fit(X_train, y_train)
##
### Get the best model
##best_rf = grid_search.best_estimator_
##
### Evaluate the model
##y_pred = best_rf.predict(X_test)
##accuracy = accuracy_score(y_test, y_pred)
##print(f'Best rf Model Accuracy: {accuracy}')
##print(f'Best Parameters: {grid_search.best_params_}')
##







##


##y_pred_rf = rf.predict(X_test)
##y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
##
### Plot AUC
##plot_auc(y_test, y_pred_proba_rf, "Random Forest")
##
### Plot Confusion Matrix
##conf(y_test, y_pred_rf, "Random Forest")



print(time()-t0)
