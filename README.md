
# Titanic Data Analysis and Prediction

This project provides several functionalities to analyze the Titanic dataset and predict the survival of passengers using a Random Forest Classifier. The functionalities include viewing the dataset, preprocessing the data, understanding correlations, evaluating the AI model, and predicting survival for new passenger data.

## Setup

Ensure you have the following dependencies installed:
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

You can install them using pip:
```
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Usage

Run the script using Python from the command line. The script supports several command-line arguments to perform different tasks. 

### Command-Line Arguments

- `1`: Display the first five rows of the Titanic dataset.
- `2`: Convert categorical values (Sex and Embarked) to numerical values.
- `3`: Fill missing values in Age, Fare, and Embarked columns with their respective means.
- `4`: Display the correlation matrix and heatmap of the dataset.
- `5`: Train a Random Forest Classifier and evaluate its accuracy.
- `6`: Predict the survival of a predefined passenger.
- `7`: Predict the survival of a passenger with details input by the user.

### Instructions

1. **Viewing the First Five Rows**
   ```
   python main.py 1
   ```

2. **Convert Categorical Values to Numerical Values**
   ```
   python main.py 2
   ```

3. **Fill Missing Values**
   ```
   python main.py 3
   ```

4. **Display Correlation Matrix and Heatmap**
   ```
   python main.py 4
   ```

5. **Evaluate AI Model**
   ```
   python main.py 5
   ```

6. **Predict Survival for a Predefined Passenger**
   ```
   python main.py 6
   ```

7. **Predict Survival for a User-Input Passenger**
   ```
   python main.py 7
   ```

### Detailed Explanation

#### 1. Viewing the First Five Rows
Displays the first five rows of the Titanic dataset to give an overview of the data structure.

#### 2. Convert Categorical Values to Numerical Values
Converts the 'Sex' column to 0 (male) and 1 (female), and the 'Embarked' column to 0 (C), 1 (Q), and 2 (S).

#### 3. Fill Missing Values
Fills missing values in the 'Age', 'Fare', and 'Embarked' columns with the mean values of their respective columns.

#### 4. Display Correlation Matrix and Heatmap
Displays the correlation matrix and a heatmap to visualize the relationships between different columns in the dataset.

#### 5. Evaluate AI Model
Trains a Random Forest Classifier on the dataset and evaluates its accuracy using accuracy score and classification report.

#### 6. Predict Survival for a Predefined Passenger
Uses the trained model to predict the survival of a predefined passenger with the following details:
- PassengerId: 172
- Pclass: 3
- Name: Rice, Master. Arthur
- Sex: male
- Age: 4
- SibSp: 4
- Parch: 1
- Ticket: 382652
- Fare: 29.125
- Cabin: NaN
- Embarked: Q

#### 7. Predict Survival for a User-Input Passenger
Prompts the user to input the details of a new passenger and predicts their survival using the trained model.

### Example

To view the first five rows of the dataset:
```
python main.py 1
```

To predict the survival of a passenger with user-input details:
```
python main.py 7
```

You will be prompted to enter details for the new passenger.

## Note

Ensure the Titanic dataset (`titanic.csv`) is located at `C:\\Users\\walid\\OneDrive\\Desktop\\titanic--main\\titanic--main\\titanic.csv` or modify the path in the script accordingly.

## Contact

For any issues or inquiries, please contact walid.ajbar@student-cs.fr and/or arthur.de-leusse@student-cs.fr

