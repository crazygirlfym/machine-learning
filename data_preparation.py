#--*- coding:utf-8 -*--
import pandas as pd
import numpy as np

#Load data:
train = pd.read_csv('Train_nyOWmfK.csv')
test = pd.read_csv('Test_bCtAN1w.csv')
print train.shape, test.shape
print train.dtypes


"""
City variable dropped because of too many categories
DOB converted to Age | DOB dropped
EMI_Loan_Submitted_Missing created which is 1 if EMI_Loan_Submitted was missing else 0 | Original variable EMI_Loan_Submitted dropped
EmployerName dropped because of too many categories
Existing_EMI imputed with 0 (median) since only 111 values were missing
Interest_Rate_Missing created which is 1 if Interest_Rate was missing else 0 | Original variable Interest_Rate dropped
Lead_Creation_Date dropped because made little intuitive impact on outcome
Loan_Amount_Applied, Loan_Tenure_Applied imputed with median values
Loan_Amount_Submitted_Missing created which is 1 if Loan_Amount_Submitted was missing else 0 | Original variable Loan_Amount_Submitted dropped
Loan_Tenure_Submitted_Missing created which is 1 if Loan_Tenure_Submitted was missing else 0 | Original variable Loan_Tenure_Submitted dropped
LoggedIn, Salary_Account dropped
Processing_Fee_Missing created which is 1 if Processing_Fee was missing else 0 | Original variable Processing_Fee dropped
Source – top 2 kept as is and all others combined into different category
Numerical and One-Hot-Coding performed
"""



#Combine into data:
train['source']= 'train'
test['source'] = 'test'
data=pd.concat([train, test],ignore_index=True)
print data.shape

# checking missing
print ("-------------check missing--------------")
print (data.apply(lambda x: sum(x.isnull())))

## look at categories of all object variables
var = ['Gender','Salary_Account','Mobile_Verified','Var1','Filled_Form','Device_Type','Var2','Source']
for v in var:
    print '\nFrequency count for variable %s'%v
    print data[v].value_counts()


## handle individual variables
print(len(data['City'].unique()))
#drop city because too many unique
data.drop('City', axis=1, inplace=True)

data['DOB'].head()

#Create age variable:
data['Age'] = data['DOB'].apply(lambda x: 115 - int(x[-2:]))
data['Age'].head()

#drop DOB:
data.drop('DOB',axis=1,inplace=True)
data.boxplot(column=['EMI_Loan_Submitted'],return_type='axes')

data['EMI_Loan_Submitted_Missing'] = data['EMI_Loan_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
print data[['EMI_Loan_Submitted','EMI_Loan_Submitted_Missing']].head(10)


#drop original vaiables:
data.drop('EMI_Loan_Submitted',axis=1,inplace=True)

data.drop('Employer_Name',axis=1,inplace=True)

print data['Existing_EMI'].describe()
#Impute by median (0) because just 111 missing:
data['Existing_EMI'].fillna(0, inplace=True)


#Majority values missing so I'll create a new variable stating whether this is missing or note:
data['Interest_Rate_Missing'] = data['Interest_Rate'].apply(lambda x: 1 if pd.isnull(x) else 0)
print data[['Interest_Rate','Interest_Rate_Missing']].head(10)
data.drop('Interest_Rate',axis=1,inplace=True)

#Drop this variable because doesn't appear to affect much intuitively
data.drop('Lead_Creation_Date',axis=1,inplace=True)


#Impute with median because only 111 missing:
data['Loan_Amount_Applied'].fillna(data['Loan_Amount_Applied'].median(),inplace=True)
data['Loan_Tenure_Applied'].fillna(data['Loan_Tenure_Applied'].median(),inplace=True)


#High proportion missing so create a new var whether present or not
data['Loan_Amount_Submitted_Missing'] = data['Loan_Amount_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)
data['Loan_Tenure_Submitted_Missing'] = data['Loan_Tenure_Submitted'].apply(lambda x: 1 if pd.isnull(x) else 0)


#Remove old vars
data.drop(['Loan_Amount_Submitted','Loan_Tenure_Submitted'],axis=1,inplace=True)

data.drop('LoggedIn',axis=1,inplace=True)

#Salary account has mnay banks which have to be manually grouped
data.drop('Salary_Account',axis=1,inplace=True)


#High proportion missing so create a new var whether present or not
data['Processing_Fee_Missing'] = data['Processing_Fee'].apply(lambda x: 1 if pd.isnull(x) else 0)
#drop old
data.drop('Processing_Fee',axis=1,inplace=True)


data['Source'] = data['Source'].apply(lambda x: 'others' if x not in ['S122','S133'] else x)
data['Source'].value_counts()


data.apply(lambda x: sum(x.isnull()))


## Numberical Coding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['Device_Type','Filled_Form','Gender','Var1','Var2','Mobile_Verified','Source']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])

# One-Hot Coding
data = pd.get_dummies(data, columns=var_to_encode)
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

train.drop('source',axis=1,inplace=True)
test.drop(['source','Disbursed'],axis=1,inplace=True)

train.to_csv('train_modified.csv',index=False)
test.to_csv('test_modified.csv',index=False)