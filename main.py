import pandas as pd
import numpy as np

originalData =  pd.read_csv('Employee Turnover Dataset.csv')
df = pd.DataFrame(originalData)

# remove whitespace from the columnames of a DataFrame
df.columns = df.columns.str.strip()
print('Print data')
print(df)

# check for duplicates
print('Check for duplicates')
print(df.duplicated())
duplicates = df.duplicated().sum()
print('Number of duplicates: ', duplicates)

# remove duplicates
df=df.drop_duplicates();

# check for missing values
missingValues = df.isnull().sum()
print('Number of missing values: ', missingValues)

#print(df.columns)
print('Check data types')
print(df.dtypes)


# code that counts how many rows in the 'HourlyRate' column contain invalid characters
# (anything other than digits, a decimal point, or a hyphen).
countInvalidValues = df['HourlyRate'].str.contains(r'[^0-9\.\-]', regex=True).sum()
# remove '$' sign
df['HourlyRate'] = df['HourlyRate'].replace({'\$': '', ',': ''}, regex=True)
# convert column to numeric
df['HourlyRate'] = pd.to_numeric(df['HourlyRate'])
# check if data type is correct
print('HourlyRate data type: ',df['HourlyRate'].dtype)

# handling missing values for categorical data by adding most frequent category
df['Turnover'].fillna(df['Turnover'].mode()[0], inplace=True)
df['CompensationType'].fillna(df['CompensationType'].mode()[0], inplace=True)
df['JobRoleArea'].fillna(df['JobRoleArea'].mode()[0], inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['MaritalStatus'].fillna(df['MaritalStatus'].mode()[0], inplace=True)
df['PaycheckMethod'].fillna(df['PaycheckMethod'].mode()[0], inplace=True)
df['TextMessageOptIn'].fillna(df['TextMessageOptIn'].mode()[0], inplace=True)

# handling missing values for numerical data by replace missing value with mean value
df['EmployeeNumber'].fillna(df['EmployeeNumber'].mean(), inplace=True)
df['HourlyRate'].fillna(df['HourlyRate'].mean(), inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Tenure'].fillna(df['Tenure'].mean(), inplace=True)
df['HoursWeekly'].fillna(df['HoursWeekly'].mean(), inplace=True)
df['AnnualSalary'].fillna(df['AnnualSalary'].mean(), inplace=True)
df['DrivingCommuterDistance'].fillna(df['DrivingCommuterDistance'].mean(), inplace=True)
df['NumCompaniesPreviouslyWorked'].fillna(df['NumCompaniesPreviouslyWorked'].mean(), inplace=True)
df['AnnualProfessionalDevHrs'].fillna(df['AnnualProfessionalDevHrs'].mean(), inplace=True)

# check for inconsistent entries - code below will display the unique values in each categorical column.
for column in df.select_dtypes(include=['object']).columns:
    print(f"Unique values in {column}: {df[column].unique()}")

# fix inconstency
df['Gender'] = df['Gender'].str.lower()  # Convert all entries to lowercase
df['CompensationType'] = df['CompensationType'].str.capitalize()  # Capitalize first letter
df['MaritalStatus'] = df['MaritalStatus'].str.capitalize()  # Capitalize first letter


# function that checks for outliers
# IQR: Any value below Q1 - 1.5IQR or above Q3 + 1.5IQR is considered an outlier.
def check_outliers(df):
    outliers = {}
    for column in df.select_dtypes(include=['int64', 'float64']).columns:
        # Using IQR to detect outliers
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        outliers[column] = df[column][outlier_condition]
    return outliers

print("5. Outliers:")
print("Outliers found in numeric columns:")
for column, outlier_values in check_outliers(df).items():
    print(f"{column}: {outlier_values}")
print("\n")


# export cleaned data
df.to_csv("CleanData.csv", index=False)