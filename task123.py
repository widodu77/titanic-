import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Task 1 


def task1():
    df = pd.read_csv('titanic.csv')
    print(df.head())

# Task 2 

# a
def task2a():   
    df = pd.read_csv('titanic.csv')

    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})

    print(df[['Sex', 'Embarked']])



# b
def task2b():

    df = pd.read_csv('titanic.csv')

    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})

    df = df.fillna({'Age': df['Age'].mean(), 'Fare': df['Fare'].mean(), 'Embarked': df['Embarked'].mean()})


    print(df[['Age', 'Fare', 'Embarked']])

# Task 3

# a
def task3a():

    df = pd.read_csv('titanic.csv')

    df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})

    df = df.fillna({'Age': df['Age'].mean(), 'Fare': df['Fare'].mean(), 'Embarked': df['Embarked'].mean()})



    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# b
    correlation_matrix = df.corr()

# c
    print(correlation_matrix)


    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()

"""
Correlation analysis :

The passenger feature is correlate dto nothing,

The survived feature ius correlatred with class and sex and is slightly xcorrelated with fare and embarked

The class feature is correlated with fare and and age and is slightly correlated with sex and embarked

The sex feature is slightly correlated with fare, Sibsp and Parch*

The Sib sp is correlated with parch and slightly correlated with fare and age



"""