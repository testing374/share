import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("train_update.csv")

##### Prepocessing
df['Sex'] = df['Sex'].map({'male':1,'female':0})
# one-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

df['Age'].fillna(df['Age'].median(), inplace=True)

# from Cabin extract Floor:int
floor_map = {'A':7,'B':6,'C':5,'D':4,'E': 3,'F': 2,'G':1}
df['Floor'] = df['Cabin'].apply(lambda x: 0 if not x or pd.isna(x) else floor_map.get(x[0],0))

df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

######

X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

########

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')

predictions = model.predict(X_test)
print(predictions)

from sklearn.metrics import classification_report
print(classification_report(y_test, (predictions > 0.5).astype(int)))
