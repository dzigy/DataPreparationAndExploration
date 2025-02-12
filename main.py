import pandas as pd
import numpy as np


originalData =  pd.read_csv('Employee Turnover Dataset.csv')
df = pd.DataFrame(originalData)

# check data type for each column
print('Check data types')
print(df.dtypes)

# remove whitespace from the column names of a DataFrame
df.columns = df.columns.str.strip()
print('Print data')
print(df)

################## DUPLICATES###################
# check for duplicates
print('Check for duplicates')
print(df.duplicated())
duplicates = df.duplicated().sum()
print('Number of duplicate entries: ', duplicates, 'rows')


# remove duplicates
df=df.drop_duplicates();
#################################################
################## MISSING VALUES################
# check for missing values
missingValues = df.isnull().sum()
print('Number of missing values per column: ')
print(missingValues)
print("\n")


# Handling missing values
# for categorical data - adding most frequent category
# for numerical data - replace missing value with median value
for column in df.columns:
    if df[column].dtype == 'object':  # Categorical columns
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:  # Numeric columns
        df[column].fillna(df[column].median(), inplace=True)

#################################################
############ INCONSISTENT ENTRIES################

# check for inconsistent entries - code below will display the unique values in each categorical column.
for column in df.select_dtypes(include=['object']).columns:
    print(f"Unique values in {column}: {df[column].unique()}")

# fix inconstency
df['Gender'] = df['Gender'].str.lower()  # Convert all entries to lowercase
df['CompensationType'] = df['CompensationType'].str.capitalize()  # Capitalize first letter
df['MaritalStatus'] = df['MaritalStatus'].str.capitalize()  # Capitalize first letter

#################################################
############ FORMATTING ERRORS ##################

# code that counts how many rows in the 'HourlyRate' column contain invalid characters
# (anything other than digits, a decimal point, or a hyphen).
countInvalidValues = df['HourlyRate'].str.contains(r'[^0-9\.\-]', regex=True).sum()
# remove '$' sign
df['HourlyRate'] = df['HourlyRate'].replace({'\$': '', ',': ''}, regex=True)
# convert column to numeric
df['HourlyRate'] = pd.to_numeric(df['HourlyRate'])
# check if data type is correct
print('HourlyRate data type: ',df['HourlyRate'].dtype)

#################################################
################## OUTLIERS #####################

# Handling outliers - using IQR to remove extreme outliers
#Using the IQR method, we calculate the lower and upper bounds and flag any values outside this range as outliers.
outliers = {}
for column in df.select_dtypes(include=['int64', 'float64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
        outliers[column] = df[column][outlier_condition]
        # capping outliers
        df[column] = df[column].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)


# outliers are valid just very different values so i am not removing them
#################################################
########### EXPORT CLEANED DATASET ##############

# export cleaned data
df.to_csv("CleanData.csv", index=False)