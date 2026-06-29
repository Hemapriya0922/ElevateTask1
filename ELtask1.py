import pandas as pd

df = pd.read_csv("titanic dataset folder/Titanic-Dataset.csv")

print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Remove Cabin column because it has many missing values
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())
from sklearn.preprocessing import LabelEncoder

# Convert Sex column into numerical values
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Convert Embarked column using one-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nDataset After Encoding:")
print(df.head())
from sklearn.preprocessing import StandardScaler

# Create scaler object
scaler = StandardScaler()

# Standardize Age and Fare columns
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\nAfter Standardization:")
print(df[['Age', 'Fare']].head())
import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot for Age
plt.figure(figsize=(6,4))
sns.boxplot(y=df['Age'])
plt.title("Boxplot of Age")
plt.show()

# Boxplot for Fare
plt.figure(figsize=(6,4))
sns.boxplot(y=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()
