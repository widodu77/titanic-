import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

path = 'titanic.csv'
# Task 1: Display the first five rows of the Titanic dataset
def task1():
    df = pd.read_csv(path)
    print(df.head())

# Task 2

# a: Convert categorical values (Sex and Embarked) to numerical values
def task2a():
    df = pd.read_csv(path)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    print(df[['Sex', 'Embarked']].head())

# b: Fill missing values in Age, Fare, and Embarked columns with their respective means
def task2b():
    df = pd.read_csv(path)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df = df.fillna({'Age': df['Age'].mean(), 'Fare': df['Fare'].mean(), 'Embarked': df['Embarked'].mean()})
    print(df[['Age', 'Fare', 'Embarked']].head())

# Task 3

# a: Display correlation matrix and heatmap
def task3a():
    df = pd.read_csv(path)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df = df.fillna({'Age': df['Age'].mean(), 'Fare': df['Fare'].mean(), 'Embarked': df['Embarked'].mean()})
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    
    # b: Calculate the correlation matrix
    correlation_matrix = df.corr()
    
    # c: Print the correlation matrix
    print(correlation_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

# Function to train a Random Forest Classifier and evaluate its performance
def forest():
    df = pd.read_csv(path)
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
def prediction_new_passenger(data):
    df = pd.read_csv(path)
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
    

    def new_info_processing(data): 

        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        data = data.fillna({'Age': df['Age'].mean(), 'Fare': df['Fare'].mean(), 'Embarked': df['Embarked'].mean()})
        data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
        return data
    

    # Preprocess the new passenger data
    new_passenger_processed = new_info_processing(data)

    # Predict survival for the new passenger
    prediction = rf.predict(new_passenger_processed)

    print(f"Prediction for the new passenger (PassengerId: {new_passenger['PassengerId'][0]}): {'Survived' if prediction[0] == 1 else 'Did not survive'}")


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

#prediction_new_passenger(new_passenger)