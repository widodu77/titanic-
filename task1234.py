import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Task 1
def task1():
    df = pd.read_csv('titanic.csv')
    print(df.head())

# Task 2

# a
def task2a():
    df = pd.read_csv('titanic.csv')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    print(df[['Sex', 'Embarked']].head())

# b
def task2b():
    df = pd.read_csv('titanic.csv')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df = df.fillna({'Age': df['Age'].mean(), 'Fare': df['Fare'].mean(), 'Embarked': df['Embarked'].mean()})
    print(df[['Age', 'Fare', 'Embarked']].head())

# Task 3

# a
def task3a():
    df = pd.read_csv('titanic.csv')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df = df.fillna({'Age': df['Age'].mean(), 'Fare': df['Fare'].mean(), 'Embarked': df['Embarked'].mean()})
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # b
    correlation_matrix = df.corr()
    
    # c
    print(correlation_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

def forest():
    df = pd.read_csv('titanic.csv')
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df = df.fillna({'Age': df['Age'].mean(), 'Fare': df['Fare'].mean(), 'Embarked': df['Embarked'].mean()})
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    
    

    # Define features and target variable
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=0)

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Display the shapes of the datasets
    print("Original Data Shape: {}".format(df.shape))
    print("Training Data Shape: {}".format(X_train.shape))
    print("Testing Data Shape: {}".format(X_test.shape))

    # Function to preprocess new passenger data
    def preprocess_new_passenger(data):
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        data = data.fillna({'Age': df['Age'].mean(), 'Fare': df['Fare'].mean(), 'Embarked': df['Embarked'].mean()})
        data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
        return data

    # New passenger data
    new_passenger = pd.DataFrame({
        'PassengerId': [172],
        'Pclass': [3],
        'Name': ['Rice, Master. Arthur'],
        'Sex': ['male'],
        'Age': [4],
        'SibSp': [4],
        'Parch': [1],
        'Ticket': ['382652'],
        'Fare': [29.125],
        'Cabin': [np.nan],
        'Embarked': ['Q']
    })

    # Preprocess the new passenger data
    new_passenger_processed = preprocess_new_passenger(new_passenger)
    print(new_passenger_processed)

    # Ensure columns match for prediction
    new_passenger_processed = new_passenger_processed[X_train.columns]

    # Predict survival for the new passenger
    prediction = rf.predict(new_passenger_processed)

    print(f"Prediction for the new passenger (PassengerId: {new_passenger['PassengerId'][0]}): {'Survived' if prediction[0] == 1 else 'Did not survive'}")
