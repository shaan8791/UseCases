import pandas as  pd
import numpy as np
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer
train = pd.read_csv(r'C:\Users\SHANTNU\Documents\GitHub\UseCases\train_u6lujuX_CVtuZ9i.csv')


#print(train.head())
X = train.iloc[:,:-1].values
#print(X)
Y = train.iloc[:,12]
#print(Y)
print(train.dtypes)
train.shape
train.head(5)
train.columns
train.sample(1)
#train.info

#1. Row wise missing values info
RowMissingCount = train.isnull().sum(axis=1)
RowMissingCount
RowMissingCount.value_counts(normalize ='False').plot.bar(title = 'Missing Row Count Frequency')

#2. Box Plots on all Columns. (Obviiously Only Continous)
#To create with looping all variables in.
#for i in train.columns:
#    train.boxplot(column=i)
#    plt.show()

train.boxplot(column=['ApplicantIncome'], grid=True)
train.boxplot(column=['CoapplicantIncome'], grid=True)
train.boxplot(column=['LoanAmount'], grid=True)


#3 Count of missing values in each column
columns = train.columns
total_rows = len(train)
count_missing = len(train) - train.count()
percent_missing = train.isnull().sum() * 100 / len(train)
missing_value_train = pd.DataFrame({ '# of total': total_rows,
                                    '# of missing': count_missing,
                                 '# of percent': percent_missing})
missing_value_train.sort_values('# of percent', inplace=True, ascending = False)
percent_missing
missing_value_train


#Declaring cross tabs for various variables with respect to output variable.
Gender = pd.crosstab(train['Gender'], train['Loan_Status'])
Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Loan_Amount_Term=pd.crosstab(train['Loan_Amount_Term'],train['Loan_Status'])


#4. Count of each category of categorical variables.
train['Gender'].value_counts(normalize ='True').plot.bar(title = 'Gender')
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

train['Married'].value_counts(normalize ='True').plot.bar(title = 'Married')
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

train['Dependents'].value_counts(normalize ='True').plot.bar(title = 'Dependents')
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

train['Education'].value_counts(normalize ='True').plot.bar(title = 'Education')
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

train['Self_Employed'].value_counts(normalize ='True').plot.bar(title = 'Self_Employed')
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

train['Loan_Amount_Term'].value_counts(normalize ='True').plot.bar(title = 'Loan_Amount_Term')
Loan_Amount_Term.div(Loan_Amount_Term.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

train['Credit_History'].value_counts(normalize ='True').plot.bar(title = 'Credit_History')
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

train['Property_Area'].value_counts(normalize ='True').plot.bar(title = 'Property_Area')
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

train['Loan_Status'].value_counts(normalize ='True').plot.bar(title = 'Loan_Status')

#5. #CORRELATION MATRIX (Pearson Correlation to measure how similar are 2 solutions)
#Pearson correlation coefficient is only used for Continous variable.
plt.figure(figsize=(16,12))
sns.heatmap(train.iloc[:,1:].corr(),annot=True,fmt=".2f")

#code for missing values
print(train.isnull().sum())
train = train.replace(0, np.NaN)
train['Gender']= train['Gender'].fillna(train['Gender'].mode()[0])
train['Married']= train['Married'].fillna(train['Married'].mode()[0])
train['Dependents']= train['Dependents'].fillna(train['Dependents'].mode()[0])
train['Self_Employed']= train['Self_Employed'].fillna(train['Self_Employed'].mode()[0])
train['LoanAmount']= train['LoanAmount'].fillna(train['LoanAmount'].mean())
#print(train['Gender'])
print(train.isnull().sum())


train['Self_Employed']='s'+ train['Self_Employed'].astype(str)
print(train['Self_Employed'])

#code for categorical variables(dummy)
propAreaD = pd.get_dummies(train['Property_Area'])
train = train.drop('Property_Area',axis=1)
train = train.join(propAreaD)
#print(propAreaD)

genderD = pd.get_dummies(train['Gender'])
train = train.drop('Gender',axis=1)
train = train.join(genderD)
#print(genderD)

marriedD =pd.get_dummies(train['Married'])
train = train.drop('Married',axis=1)
train = train.join(marriedD)
#print(marriedD)

educationD = pd.get_dummies(train['Education'])
train = train.drop('Education',axis=1)
train = train.join(educationD)
#print(educationD)

selfEmpD = pd.get_dummies(train['Self_Employed'])
train = train.drop('Self_Employed',axis=1)
train = train.join(selfEmpD)
#print(selfEmpD)



train.head(5)

import matplotlib
import matplotlib.pyplot as plt
train['LoanAmount'].plot(kind = 'hist',bins =50)