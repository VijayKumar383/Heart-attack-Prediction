# Heart-attack-Prediction
The Heart Attack Prediction Project aims to develop a predictive model to assess an individual's risk of experiencing a heart attack based on various health indicators and lifestyle factors. By leveraging data analysis and machine learning techniques.
# README for Health Data Analysis Project

## Overview

This project involves analyzing a health dataset to gain insights into various factors affecting health conditions and build predictive models. The analysis includes data preprocessing, exploratory data analysis (EDA), and building several machine learning models to classify health conditions.

## Contents

1. **Importing Health Data**
2. **Identifying Null Values**
3. **Examining Data Types**
4. **Identifying Numerical and Categorical Features**
5. **Converting Features to Categorical Data Types**
6. **Exploring Feature Correlations**
7. **Visualizing Health Conditions**
8. **Analyzing Health Conditions by Gender**
9. **Examining Chest Pain Types and Health Conditions**
10. **Investigating Fasting Blood Sugar Levels and Health Conditions**
11. **Analyzing Resting Electrocardiographic Results and Health Conditions**
12. **Examining Exercise-Induced Angina and Health Conditions**
13. **Investigating the Slope of the ST Segment and Health Conditions**
14. **Analyzing the Number of Major Vessels Colored by Fluoroscopy and Health Conditions**
15. **Examining Thalassemia and Health Conditions**
16. **Visualizing Age Distribution**
17. **Visualizing Resting Blood Pressure Distribution**
18. **Visualizing Cholesterol Distribution**
19. **Visualizing Maximum Heart Rate Distribution**
20. **Visualizing ST Depression Distribution**
21. **Analyzing Fasting Blood Sugar Levels and Health Conditions**
22. **Visualizing Chest Pain Types, Age, and Health Conditions**
23. **Encoding Categorical Features**
24. **Preparing Features and Target Variable**
25. **Scaling Features**
26. **Splitting the Data into Training and Testing Sets**
27. **Building and Evaluating Logistic Regression Model**
28. **Building and Evaluating Linear Discriminant Analysis Model**
29. **Building and Evaluating K-Nearest Neighbors (KNN) Model**
30. **Building and Evaluating Decision Tree Classifier Model**
31. **Building and Evaluating Gaussian Naive Bayes Model**
32. **Building and Evaluating Random Forest Classifier Model**
33. **Building and Evaluating Support Vector Classifier (SVC) Model**
34. **Evaluating Model Performance**
35. **Making Predictions with Gaussian Naive Bayes Model**
36. **Saving the Logistic Regression Model**

## Prerequisites

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Instructions

1. **Importing Health Data**

   Load the health data from a CSV file into a Pandas DataFrame.

   ```python
   import pandas as pd
   df = pd.read_csv("./heart_cleveland_upload.csv")
   ```

2. **Identifying Null Values**

   Check for any missing values in the dataset.

   ```python
   sumofnull = df.isnull().sum()
   ```

3. **Examining Data Types**

   Review the data types of each column to understand the types of data.

   ```python
   datatype = df.dtypes
   ```

4. **Identifying Numerical and Categorical Features**

   Classify features into numerical and categorical categories.

   ```python
   numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'condition']
   cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
   ```

5. **Converting Features to Categorical Data Types**

   Convert specific columns to categorical data types.

   ```python
   lst = cat_features
   df[lst] = df[lst].astype(object)
   ```

6. **Exploring Feature Correlations**

   Create a heatmap to visualize correlations between numerical features.

   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt

   selected_columns = df[numeric_features]
   corr_data = selected_columns.corr()

   plt.figure(figsize=(10, 8))
   sns.heatmap(corr_data, annot=True, cmap='RdBu', linewidths=0.1)
   plt.title('Correlation Between Numeric Features')
   plt.show()
   ```

7. **Visualizing Health Conditions**

   Plot the distribution of health conditions in the dataset.

   ```python
   condition_ax = sns.countplot(x=df["condition"], palette='bwr')
   plt.show()
   ```

8. **Analyzing Health Conditions by Gender**

   Compare the distribution of health conditions by gender.

   ```python
   sex_ax = sns.countplot(x=df["sex"], hue=df['condition'],  palette='bwr')
   plt.show()
   ```

9. **Examining Chest Pain Types and Health Conditions**

   Analyze the relationship between chest pain types and health conditions.

   ```python
   cp_ax = sns.countplot(x=df["cp"], hue=df['condition'], palette='bwr')
   plt.show()
   ```

10. **Investigating Fasting Blood Sugar Levels and Health Conditions**

    Examine the relationship between fasting blood sugar levels and health conditions.

    ```python
    fbs_ax = sns.countplot(x=df["fbs"], hue=df['condition'], palette='bwr')
    plt.show()
    ```

11. **Analyzing Resting Electrocardiographic Results and Health Conditions**

    Look at the relationship between resting electrocardiographic results and health conditions.

    ```python
    restecg_ax = sns.countplot(x=df["restecg"], hue=df['condition'], palette='bwr')
    plt.show()
    ```

12. **Examining Exercise-Induced Angina and Health Conditions**

    Analyze how exercise-induced angina relates to health conditions.

    ```python
    exang_ax = sns.countplot(x=df["exang"], hue=df['condition'], palette='bwr')
    plt.show()
    ```

13. **Investigating the Slope of the ST Segment and Health Conditions**

    Examine the relationship between the slope of the ST segment and health conditions.

    ```python
    slope_ax = sns.countplot(x=df["slope"], hue=df['condition'], palette='bwr')
    plt.show()
    ```

14. **Analyzing the Number of Major Vessels Colored by Fluoroscopy and Health Conditions**

    Analyze how the number of major vessels colored by fluoroscopy relates to health conditions.

    ```python
    ca_ax = sns.countplot(x=df["ca"], hue=df['condition'], palette='bwr')
    plt.show()
    ```

15. **Examining Thalassemia and Health Conditions**

    Investigate how different thalassemia categories relate to health conditions.

    ```python
    thal_ax = sns.countplot(x=df["thal"], hue=df['condition'], palette='bwr')
    plt.show()
    ```

16. **Visualizing Age Distribution**

    Create a histogram to visualize the distribution of age.

    ```python
    age_col = df['age']

    plt.figure(figsize=(10,6))
    plt.hist(age_col, bins=20, color='skyblue', alpha=0.7, ec='blue')

    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution')

    plt.show()
    ```

17. **Visualizing Resting Blood Pressure Distribution**

    Create a histogram to visualize the distribution of resting blood pressure.

    ```python
    trestbps_col = df['trestbps']

    plt.figure(figsize=(10, 6))
    plt.hist(trestbps_col, bins=20, color='lightcoral', alpha=0.7, ec='red')

    plt.xlabel('trestbps')
    plt.ylabel('Frequency')
    plt.title('trestbps Distribution')

    plt.show()
    ```

18. **Visualizing Cholesterol Distribution**

    Create a histogram to visualize the distribution of cholesterol levels.

    ```python
    chol_col = df['chol']

    plt.figure(figsize=(10, 6))
    plt.hist(chol_col, bins=20, color='lightgreen', alpha=0.7, ec='green')

    plt.xlabel('Cholesterol (chol)')
    plt.ylabel('Frequency')
    plt.title('Cholesterol Distribution')

    plt.show()
    ```

19. **Visualizing Maximum Heart Rate Distribution**

    Create a histogram to visualize the distribution of maximum heart rate.

    ```python
    thalach_col = df['thalach']

    plt.figure(figsize=(10, 6))
    plt.hist(thalach_col, bins=20, color='cyan', alpha=0.7, ec="darkblue")

    plt.xlabel('Maximum Heart Rate (thalach)')
    plt.ylabel('Frequency')
    plt.title('Maximum Heart Rate Distribution')

    plt.show()
    ```

20. **Visualizing ST Depression Distribution**

    Create a histogram to visualize the distribution of ST depression.

    ```python
    oldpeak_col = df['oldpeak']

    plt.figure(figsize=(10, 6))
    plt.hist(oldpeak_col, bins=20, color='orange', alpha=0.7, ec="darkred")

    plt.xlabel('ST depression')
    plt.ylabel('Frequency')
    plt.title('Distribution of ST depression')

    plt.show()
    ```

21. **Analyzing Fasting Blood Sugar Levels and Health Conditions**

    Use a countplot to examine the relationship between fasting blood sugar levels and health conditions.

    ```python
    countplt = sns.catplot(x='fbs', hue='condition', kind='count', alpha=0.85, data=df, palette='bwr')
    plt.show()
    ```

