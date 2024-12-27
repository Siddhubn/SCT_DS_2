# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure seaborn for better visuals
sns.set_theme(style="whitegrid", palette="muted")

# Load datasets
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
gender_submission_df = pd.read_csv('dataset/gender_submission.csv')

# Combine train and test datasets for consistent cleaning
test_df['Survived'] = np.nan
combined_df = pd.concat([train_df, test_df], sort=False)

# Check missing values
print("\nMissing values before cleaning:")
print(combined_df.isnull().sum())

# Fill missing values for 'Age' with median grouped by 'Pclass' and 'Sex'
combined_df['Age'] = combined_df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# Fill missing 'Embarked' with the most frequent value (mode)
combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)

# Fill missing 'Fare' with median (for test data)
combined_df['Fare'].fillna(combined_df['Fare'].median(), inplace=True)

# Drop 'Cabin' column due to excessive missing data
combined_df.drop(columns=['Cabin'], inplace=True)

# Feature Engineering: Extract Title from the Name column
combined_df['Title'] = combined_df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
combined_df['Title'] = combined_df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 
                                                     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined_df['Title'] = combined_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Mme', 'Mrs')

# Map categorical variables to numeric values
combined_df['Sex'] = combined_df['Sex'].map({'male': 1, 'female': 0})
combined_df['Embarked'] = combined_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
combined_df['Title'] = combined_df['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5})

# Drop unnecessary columns
combined_df.drop(columns=['Name', 'Ticket'], inplace=True)

# Split the combined dataset back into train and test datasets
train_df = combined_df[~combined_df['Survived'].isnull()]
test_df = combined_df[combined_df['Survived'].isnull()].drop(columns=['Survived'])

# Convert 'Survived' column to integer
train_df['Survived'] = train_df['Survived'].astype(int)

# ----- Exploratory Data Analysis (EDA) -----

# 1. Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], kde=True, bins=30, color='coral')
plt.title('Age Distribution of Passengers', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# 2. Fare Distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Fare'], kde=True, bins=30, color='purple')
plt.title('Fare Distribution of Passengers', fontsize=16)
plt.xlabel('Fare', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# 3. Survival Rate by Sex
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=train_df, palette=['lightblue', 'pink'])
plt.title('Survival Rate by Gender', fontsize=16)
plt.xlabel('Sex (0 = Female, 1 = Male)', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.show()

# 4. Survival Rate by Passenger Class
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=train_df, palette='YlGnBu')
plt.title('Survival Rate by Passenger Class', fontsize=16)
plt.xlabel('Pclass', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.show()

# 5. Survival Rate by Embarked Location
plt.figure(figsize=(8, 5))
sns.barplot(x='Embarked', y='Survived', data=train_df, palette='coolwarm')
plt.title('Survival Rate by Embarkation Port', fontsize=16)
plt.xlabel('Embarked (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.show()

# 6. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(train_df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', cbar=True, linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# 7. Pairplot for Important Features
important_features = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
sns.pairplot(train_df[important_features], hue='Survived', palette='husl', diag_kind='kde')
plt.show()

# 8. Survival Rate by Title
plt.figure(figsize=(10, 6))
sns.barplot(x='Title', y='Survived', data=train_df, palette='viridis')
plt.title('Survival Rate by Title', fontsize=16)
plt.xlabel('Title', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.show()

# 9. Survival by Age and Fare (Scatter Plot)
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train_df, palette='cool', alpha=0.8)
plt.title('Survival by Age and Fare', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Fare', fontsize=12)
plt.show()

# Summary Statistics of Important Variables
print("\nSummary Statistics:")
print(train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].describe())

# Survival Rates by Groups
print("\nSurvival Rate by Gender:")
print(train_df.groupby('Sex')['Survived'].mean())

print("\nSurvival Rate by Pclass:")
print(train_df.groupby('Pclass')['Survived'].mean())

print("\nSurvival Rate by Embarked:")
print(train_df.groupby('Embarked')['Survived'].mean())

print("\nSurvival Rate by Title:")
print(train_df.groupby('Title')['Survived'].mean())