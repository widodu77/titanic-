import task1234  # Import the module containing the task functions
import sys 
import pandas as pd
import numpy as np

# Check if no command-line arguments are provided
if len(sys.argv)==1:

    # Print welcome message and instructions for the user

    print("Dear user, hello and welcome here you will find the informations you need to make our software function")

    print("If you are interested in seeing the first five rows of the titanic file type on your shell: python main.py 1")

    print("If you are interested in modifying this dataframe by converting the sex and embarked values to numerical ones  type : python main.py 2")

    print("If you are interested in modifying this dataframe by changing all Nan values of the columns age fare and embarked by the mean of these columns  type : python main.py 3")

    print("If you are interested in understanding the correlations between the different columns  type : python main.py 4")

    print("If you are interested in understanding the accuracy of our ai model  type : python main.py 5")

    print("if you are interested in seeing a test case that predicts the survival of a passenger, type : python main.py 6")

    print("and finally, if you want to use the model to try predicting a specific case that you input, simply type: python main.py 7, and then you'll be asked to fill out the info you want to work on ")

    print("PS : PLEASE NOTE THAT EVERY INPUT (from 1 to 7) WORKS ON THEIR ON, YOU DO NOT NEED TO LOAD ANYTHING TO MAKE THE LAST TASKS WORK ")

# Example new passenger data for prediction
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

# Function to collect new passenger information from the user
def sett():
    new_passenger = pd.DataFrame({
            'PassengerId': [int(input("Enter Passenger ID: "))],
            'Pclass': [int(input("Enter Pclass (1, 2, or 3): "))],
            'Name': [input("Enter Name: ")],
            'Sex': [input("Enter Sex (male or female): ")],
            'Age': [float(input("Enter Age: "))],
            'SibSp': [int(input("Enter number of Siblings/Spouses Aboard: "))],
            'Parch': [int(input("Enter number of Parents/Children Aboard: "))],
            'Ticket': [input("Enter Ticket Number: ")],
            'Fare': [float(input("Enter Fare: "))],
            'Cabin': [input("Enter Cabin: (or leave blank if unknown) ") or np.nan],
            'Embarked': [input("Enter Port of Embarkation (C, Q, or S): ")]
        })
    return new_passenger

# Check if exactly one command-line argument is provided
if len(sys.argv)==2:
    if sys.argv[1]=='1':
        task1234.task1()

    if sys.argv[1]=='2':
        task1234.task2a()

    if sys.argv[1]=='3':
        task1234.task2b()

    if sys.argv[1]=='4':
        task1234.task3a()

    if sys.argv[1]=='5':
        task1234.forest()

    if sys.argv[1]=='6':
        task1234.prediction_new_passenger(new_passenger)

    if sys.argv[1]=='7':
        task1234.prediction_new_passenger(sett())

