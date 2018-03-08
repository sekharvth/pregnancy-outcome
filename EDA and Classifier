#import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# read the 'Mother Report' excel file
mother_report = pd.read_excel('Mother_Report.xlsx')

# drop unnecessary columns toward the end, such as 'Created By', 'Updated By', 'Device ID' etc. and also 
# 'Eligible Couple Number, Other Religion, Vdrl Test Date and HIV Test date as those dates don't significantly impact the outcome of birth 
mother_report = mother_report.drop(mother_report.columns[-1:-13:-1], axis = 1)
mother_report = mother_report.drop(['Eligible Couple Number', 'Other Religion', 'Vdrl Test Date', 'Hiv Test Date'], axis = 1)
mother_report = mother_report.drop(mother_report.columns[-1:-8:-1], axis = 1)

# EDA. To find out the amount of Missing Values in each of the columns that start with 'ANC1'.
# Same has benn done for other columns too, but not reproduced here.
mother_report[mother_report.columns[mother_report.columns.str.startswith('ANC1')]].isnull().sum()

# Read in the 'Delivery_Outcome_report file that has Birth Outcome data
delivery_outcome = pd.read_excel('Delivery_Outcome_Report.xlsx')

# Set the entries of outcome variable to 'Abortion' and 'Miscarriage' when 'No. Abortion' > 0 and 'No. Miscarriage' > 0
delivery_outcome.loc[(delivery_outcome['No. Abortion'] > 0)  , ['Live Birth Or Still Birth']] = 'Abortion'
delivery_outcome.loc[(delivery_outcome['No. Miscarriage'] > 0) , ['Live Birth Or Still Birth']] = 'Miscarriage'

#Filter out the entries that don't have a value in the 'Live Birth Or Still Birth' field and rename it as delivery_1
delivery_1 = delivery_outcome[~delivery_outcome['Live Birth Or Still Birth'].isnull()]

#Read in the other file with outcomes.
delivery_dhan = pd.read_excel('Dhanwant Delivery Call Data.xlsx')

#Set the value for entries where the outcome variable is not missing AND where the infant has died, to 'Infant Death' and name the 
#resulting dataframe as delivery_2
delivery_dhan.loc[((~delivery_dhan['Delivery outcome'].isnull()) & (delivery_dhan['Infant Death'] == 'Yes')), 'Delivery outcome'] = 'Infant Death'
delivery_2 = delivery_dhan.loc[((~delivery_dhan['Delivery outcome'].isnull()) & (delivery_dhan['Infant Death'] == 'Yes'))]

#make sure that it only contains entries whose outcome values are not missing
delivery_2 = delivery_2[~delivery_2['Delivery outcome'].isnull()]

#Now we've to make a single dataframe(df) having data from both 'Delivery_Outcome_report and Dhanwant_Delivery
#Make a temporary df having relevant columns from delivery_1
merge = pd.DataFrame()
merge['ID'] = delivery_1['Mother ID']
merge['Outcome'] = delivery_1['Live Birth Or Still Birth']

#Another temporary df that takes in all the non-common patients from delivery_1 and delivery_2
tmp = delivery_2[~delivery_2['Patient ID'].isin(delivery_1['Mother ID'])][['Patient ID', 'Delivery outcome']]


#Now we concatenate the non-common entries from both delivery_1 and delivery_2 to the 'merge' dataframe containing delivery_1 entries
#First we rename the columns in 'tmp' so that they are concatenated along the same columns names
tmp = tmp.reset_index().drop('index', axis = 1).rename(columns = {'Patient ID' : 'ID', 'Delivery outcome' : 'Outcome'})
#concatenate along the columns( add new rows with same columns )
merged = pd.concat((merge, tmp), axis = 0, ignore_index = True)

#Now merge it with the 'Mother_report' df so that we have outcomes for the data we need to train on
df = pd.merge(mother_report, merged, left_on = 'Registration ID', right_on = 'ID').drop('ID',axis = 1)

#Make sure that all outcomes ('Live Birth', 'Still Birth', 'Abortion', 'Infant Death', 'Miscarriage') are present in the final df
df['Outcome'].unique()

#Again see how many missing values are left in the new df
df[df.columns[df.columns.str.startswith('ANC1')]].isnull().sum()

#Plot a CountPlot to find out how each variable affects the outcome individually
fig, ax = plt.subplots(figsize = (20,15))
sns.countplot(df['ANC2Day'] , hue= df2.Outcome)
plt.show()

#Drop some of the following columns based on inferences from the countplot
#'Religion' dropped because the 'Caste' variable gives a much better picture than religion and there is no need for redundant variables
#The different ANC IFA dates are dropped because they are the same as the corresponding ANC dates.
#'Urine Sugar' and albumin are dropped as they appear in the 'High Risk Symptoms' variable on which some feature engg. has been done. More on that later.
# and the remaining have been dropped because they contain too many missing values to make a guesstimate.
df2 = df.drop(['LMP Date', 'Religion', 'Bank Name', 'Blood Group', 'Drop Out',
         'Past Illnesses', 'Penultimate Preg Outcomes', 
         'Prior Pregnancy Complications', 'Block', 'Camp Name', 
         'Active', 'Returning', 
         'ANC1 IFA Date', 'ANC2 IFA Date', 'ANC3 IFA Date', 'ANC4 IFA Date',
         'ANC1 Albendazole Status', 'ANC2 Albendazole Status', 'ANC3 Albendazole Status', 'ANC4 Albendazole Status',
         'ANC1 Referral Location', 'ANC2 Referral Location', 'ANC3 Referral Location', 'ANC4 Referral Location',
        'ANC1 Albendazole', 'ANC2 Albendazole', 'ANC3 Albendazole', 'ANC4 Albendazole', 
         'ANC1 IFA Status', 'ANC2 IFA Status', 'ANC3 IFA Status', 'ANC4 IFA Status', 
         'ANC1 Urine Sugar', 'ANC2 Urine Albumin', 'ANC2 Urine Sugar',"ANC1 Urine Albumin ",
        'ANC3 Urine Albumin', 'ANC3 Urine Sugar', 'ANC4 Urine Albumin', 'ANC4 Urine Sugar'], axis = 1)
		
# Since the sting values in the data such as 'No Equipment' in the variable 'ANC Heart Rate' have been misspelled 
#and expressed in different languages, make it into a single value, 'NO Equipment'
df2[df2.columns[df2.columns.str.contains('Heart Rate')]].loc [((df2['ANC1 Fetal Heart Rate'].str.contains('quipm')) | (df2['ANC2 Fetal Heart Rate'].str.contains('quipm'))
                         | (df2['ANC3 Fetal Heart Rate'].str.contains('quipm')) | (df2['ANC4 Fetal Heart Rate'].str.contains('quipm')))].fillna('No Equipment', inplace = True)
						 
# Fill in the missing values in the 'ANC Dates' columns to -1 
for i in [1,2,3,4]:
    df2['ANC'+str(i)+' Date'] = df2['ANC'+str(i)+' Date'].map(lambda x: x if x.year>1970 else -1)
	
# Fill in the missing values in ANC Weights with NaN values for the time being (not -1, as next we find out the diff in weights across all ANC's)
for i in ['ANC1 Weight', 'ANC2 Weight', 'ANC3 Weight', 'ANC4 Weight']:
    df2[i] = df2[i].map(lambda x: x if (type(x) == int or type(x) == float) else np.nan)

# Normally the weights tend to increase as the ANC's progress. Although some entries buck this trend, this trend has been assumed here
# to find out the max weight change from the first ANC. If all weight values are missing, set weight change to 0
df2['Max Change in ANC Weight'] = df2[['ANC1 Weight', 'ANC2 Weight', 'ANC3 Weight', 'ANC4 Weight']].apply(lambda x : x.max() - x['ANC1 Weight'] if x['ANC1 Weight'] != -1 else 0, axis = 1)

# Drop the columns as data has been extracted
df2.drop(['ANC1 Weight', 'ANC2 Weight', 'ANC3 Weight', 'ANC4 Weight'], axis = 1, inplace = True)

# Find out the pressure differential (('Sys BP' - 'Dia BP)/'Sys BP') that takes into account the absolute values of the 'Sys BP',
# rather than just the difference in pressure. This is done on the huge dataframe where the large no. of columns
# were not dropped. 
df['ANC4 Pressure Difference'] = df.apply(lambda x : (x['ANC4 BP Sys'] - x['ANC4 BP Dia'])/x['ANC4 BP Sys'] if (type(x['ANC4 BP Sys']) == int and x['ANC4 BP Sys' ] != 0) else 0, axis = 1)
df['ANC3 Pressure Difference'] = df.apply(lambda x : (x['ANC3 BP Sys'] - x['ANC3 BP Dia'])/x['ANC3 BP Sys'] if (type(x['ANC3 BP Sys']) == int and x['ANC3 BP Sys' ] != 0) else 0, axis = 1)
df['ANC2 Pressure Difference'] = df.apply(lambda x : (x['ANC2 BP Sys '] - x['ANC2 BP Dia'])/x['ANC2 BP Sys '] if (type(x['ANC2 BP Sys ']) == int and x['ANC2 BP Sys ' ] != 0) else 0, axis = 1)
df['ANC1 Pressure Difference'] = df.apply(lambda x : (x['ANC1 BP Sys'] - x['ANC1 BP Dia'])/x['ANC1 BP Sys'] if (type(x['ANC1 BP Sys']) == int and x['ANC1 BP Sys' ] != 0) else 0, axis = 1)

# make those values into new columns in our main dataframe replacing -1 values, if any, to 0 
for i in ['ANC1 Pressure Difference', 'ANC2 Pressure Difference', 'ANC3 Pressure Difference', 'ANC4 Pressure Difference']:
    df2[i] = df[i].map(lambda x: 0 if x == -1 else x)
	
# Drop the BP columns as relevant data has been extracted from it.
df2.drop(df2.columns[df2.columns.str.contains('BP')], axis = 1, inplace = True)

# Keep only the ANC1 Hb values as all the other contain too much missing data and fill missing values in ANC1 Hb to -1
df2.drop(['ANC2 HB', 'ANC3 HB', 'ANC4 HB'], axis = 1, inplace = True)
df2['ANC1 HB'] = df2['ANC1 HB'].map(lambda x: -1 if type(x) == str else x).fillna(-1)

# Drop the fundal height valriables as they appear in the risk symptoms engineered features later on.
df2.drop(['ANC1 Fundal Height', 'ANC2 Fundal Height', 'ANC3 Fundal Height', 'ANC4 Fundal Height'], axis = 1, inplace = True)

# Assign values for 'Abnormal', 'Normal', and 'Not Done' values in ANC fetal movements variable 
for i in ['ANC1 Fetal Movements', 'ANC2 Fetal Movements', 'ANC3 Fetal Movements', 'ANC4 Fetal Movements']:
    df2[i] = df[i].map({'Abnormal' : -100, 'Normal' : 100, 'Not Done' : -1})

# Here, we find out the max values (100) and min values (-100) along the different Fetal mov variables in one entry, temporary feature.
df2['max'] = df2[['ANC1 Fetal Movements', 'ANC2 Fetal Movements', 'ANC3 Fetal Movements', 'ANC4 Fetal Movements']].max(axis= 1)
df2['min'] = df2[['ANC1 Fetal Movements', 'ANC2 Fetal Movements', 'ANC3 Fetal Movements', 'ANC4 Fetal Movements']].min(axis= 1)
 
# Create a new feature that shows whether fetal mov was normal or abnormal
# If along the row, the max value of 100 is present, then fetal mov was normal in atleast one ANC checkup, so assign 1 to it (normal)
# Else assign zero (abnormal)
df2['Fetal Movement Nomal/Abnormal'] = df2.apply(lambda x: 1 if x['max'] == 100 else 0, axis = 1)

# If both max and min are -1 (missing), assign to the new feature a value of 2, indicating 'Unknown' 
df2.loc[((df2['max'] == -1) & (df2['min'] == -1)), 'Fetal Movement Nomal/Abnormal'] = 2

#Drop temp columns max and min
df2.drop(['max', 'min'], axis = 1, inplace = True)

# Do the same for Fetal Position variables
for i in ['ANC1 Fetal Position', 'ANC2 Fetal Position', 'ANC3 Fetal Position', 'ANC4 Fetal Position']:
    df2[i] = df[i].map({'Abnormal' : -100, 'Normal' : 100, 'Not Done' : -1})
df2['max'] = df2[['ANC1 Fetal Position', 'ANC2 Fetal Position', 'ANC3 Fetal Position', 'ANC4 Fetal Position']].max(axis= 1)
df2['min'] = df2[['ANC1 Fetal Position', 'ANC2 Fetal Position', 'ANC3 Fetal Position', 'ANC4 Fetal Position']].min(axis= 1)
df2['Fetal Position Nomal/Abnormal'] = df2.apply(lambda x: 1 if x['max'] == 100 else 0, axis = 1)
df2.loc[((df2['max'] == -1) & (df2['min'] == -1)), 'Fetal Position Nomal/Abnormal'] = 2
df2.drop(['max', 'min'], axis = 1, inplace = True)

# After extracting relevant data, delete the columns.
df2.drop(['ANC1 Fetal Movements', 'ANC2 Fetal Movements', 'ANC3 Fetal Movements', 'ANC4 Fetal Movements'], axis = 1, inplace = True)
df2.drop(['ANC1 Fetal Position', 'ANC2 Fetal Position', 'ANC3 Fetal Position', 'ANC4 Fetal Position'], axis = 1, inplace = True)

# Assign the missing values in 'Heart Rate' a value of -1
for i in ['ANC1 Fetal Heart Rate', 'ANC2 Fetal Heart Rate', 'ANC3 Fetal Heart Rate', 'ANC4 Fetal Heart Rate']:
    df2[i] = df2[i].map(lambda x: -1 if type(x) != int else x)

df2.drop('ANC1 Fetal Heart Rate', axis = 1, inplace =True) # only 40 real values, so drop it

# Across all the 'ANC HIgh Risk Symptoms', fill in missing values with 'none' and make everything lowercase
df2['ANC1 High Risk Symptoms'] = df2['ANC1 High Risk Symptoms'].map(lambda x : 'none' if type(x) != str else x.lower())
df2['ANC2 High Risk Symptoms'] = df2['ANC2 High Risk Symptoms'].map(lambda x : 'none' if type(x) != str else x.lower())
df2['ANC3 High Risk Symptoms'] = df2['ANC3 High Risk Symptoms'].map(lambda x : 'none' if type(x) != str else x.lower())
df2['ANC4 High Risk Symptoms'] = df2['ANC4 High Risk Symptoms'].map(lambda x : 'none' if type(x) != str else x.lower())

# Make a temp column that has all the different symptoms across all ANC's. Here duplicates along all the checkups
# have been deliberately kept, as that shows the severity of the symptom and non treatment of the same
df2['temp'] = list(map(lambda x: x.lstrip(), df2[['ANC1 High Risk Symptoms', 'ANC2 High Risk Symptoms', 'ANC3 High Risk Symptoms', 'ANC4 High Risk Symptoms']]\
.apply(lambda x: (', ').join([x['ANC1 High Risk Symptoms'], x['ANC2 High Risk Symptoms'], x['ANC3 High Risk Symptoms'], \
                             x['ANC4 High Risk Symptoms']]), axis = 1)))


# Since the number of unique syptoms amount to more than 100, we find out the most frequently occuring 14 symptoms
# make a list of out of all the symptoms simillar to 'temp' column
lst = list(map(lambda x: x.lstrip(), (',').join(df2['ANC2 High Risk Symptoms']).split(','))) + list(map(lambda x: x.lstrip(), (',').join(df2['ANC1 High Risk Symptoms']).split(','))) + \
        list(map(lambda x: x.lstrip(), (',').join(df2['ANC3 High Risk Symptoms']).split(','))) + list(map(lambda x: x.lstrip(), (',').join(df2['ANC4 High Risk Symptoms']).split(',')))
# Dictionary to store symptoms and their counts
dic = {}
# Get count of each symptom
for i in lst:
    dic[i] = dic.get(i, 0) + 1
# Sort the dictionary in descending order of counts and take the top 14 entries and store it in risks_final list
import operator
risks_final = [i for i,j in sorted(dic.items(), key = operator.itemgetter(1), reverse = True)[1:15]]
sorted(dic.items(), key = operator.itemgetter(1), reverse  = True)[1:25]

# make a new column for each risk and indicate presence of the risk in the entry with 1 or 0
for i in df2.index:
    for risk in risks_final:
        if risk in df2.loc[i, 'temp']:
            df2.loc[i, risk] = 1
        else:
            df2.loc[i, risk] = 0
            
# Some feature engg. that finds out the no. of conditions(risky symptoms),the total no. of characters, 
# and the total no. of words across 4 ANC's,
df2['No. of conditions over 4 ANC\'s'] = df2['temp'].map(lambda x: len([i for i in x.split(', ') if i != 'none']))
df2['No. of characters over 4 ANC\'s risk symptoms'] = df2['temp'].map(lambda x: len(x) if 'none' not in x else sum([len(i) for i in x.split(', ') if i!= 'none']) )
df2['No. of words in risk symptoms over 4 ANC\'s' ] = df2['temp'].map(lambda x: len([i for i in x.split(' ') if i != 'none' and i != 'none,']))

# drop the high risk columns as data has been extracted from it.
df2.drop(['ANC1 High Risk Symptoms', 'ANC2 High Risk Symptoms', 'ANC3 High Risk Symptoms', 'ANC4 High Risk Symptoms'], axis = 1, inplace = True)

# Drop the blood sugar status columns columns as they are already present as 'diabetes' in the symptoms features
df2.drop(df2.columns[df2.columns.str.startswith('Blood')], axis = 1, inplace = True)

# Fill missing values in below columns to 0 ( 0 used because the actual entries in the columns are marked NA (not applicable) instead of being missing)
df2['FA Tablets given before 12 weeks (number)'] = df2['FA Tablets given before 12 weeks (number)'].map(lambda x: 0 if type(x) != int else x)
df2['IFA Tablets given after 12 weeks (number)'] = df2['IFA Tablets given after 12 weeks (number)'].map(lambda x: 0 if type(x) != int else x)

# If there is a string value in below column such as 'Mother Denies', make it 0
df2.loc[df2['Albendazole Tablets Given(number)'].map(lambda x: type(x) == str), 'Albendazole Tablets Given(number)'] = 0

# Fill missing values with -1
df2['Albendazole Tablets Given(number)'] = df2['Albendazole Tablets Given(number)'].fillna(-1)

# Drop the blood sugar values across the ANC's
df2.drop(['ANC1 Blood Sugar', 'ANC2 Blood Sugar', 'ANC3 Blood Sugar', 'ANC4 Blood Sugar'], axis = 1, inplace = True)

# Now that we have completed EDA and feature engineering, find out how many missing values are left
df2.isnull().sum()

# Fill the only missing value in 'Age' with the mean of the age 
df2.Age.fillna(int(df2.Age.mean()), inplace = True)

# Fill missing height values with the median
df2.Height.fillna(int(df2.Height.median()), inplace = True)

# The most frequently occuring caste is 'ST', so fill missing values accordingly
df2.Caste.fillna('ST', inplace = True)

# Fill the missing values in Last Pregnancy Outcome to 'Unknown' 
df2['Last Pregnancy Outcome'].fillna('Unknown', inplace = True)

#  To fill missing values in 'Max Change in ANC Weight':
# remove outliers in diff in weights and set missing values to mean
avg = np.mean(df2[(df2['Max Change in ANC Weight'] != 40.8) & (df2['Max Change in ANC Weight'] != 36.8 ) & (df2['Max Change in ANC Weight'] != 140)]['Max Change in ANC Weight'])
df2['Max Change in ANC Weight'].fillna(avg, inplace = True)
df2.loc[(df2['Max Change in ANC Weight'] == 40.8) | (df2['Max Change in ANC Weight'] == 36.8 ) | (df2['Max Change in ANC Weight'] == 140), \
'Max Change in ANC Weight'] = avg

# Set the values expressed in Hindi in the below column to 'Given' and fill missing values with 'Unknown'
df2.loc[(df2['TT1 Given/Denied Status'] ==  'ttbuster diya') | (df2['TT1 Given/Denied Status'] == 'TT buster diya'), 'TT1 Given/Denied Status'] = 'Given'
df2.loc[((df2['TT2 Given/Denied Status'] == 'booster diya gaya') | (df2['TT2 Given/Denied Status'] == 'pichli baar de diya') |(df2['TT2 Given/Denied Status'] == 'Avoid wastage')  |(df2['TT2 Given/Denied Status'] == 'TT 1 hi booster he jo diya gaya ' ) | (df2['TT2 Given/Denied Status'] == 'TT buster  diya 1st anc me') |(df2['TT2 Given/Denied Status'] == 'booster') | (df2['TT2 Given/Denied Status'] == 'buster diya' ) | (df2['TT2 Given/Denied Status'] == 'buster dea' ) |(df2['TT2 Given/Denied Status'] == 'T T buster lagaya') | (df2['TT2 Given/Denied Status'] == 'TT=1 hi booster hai') |(df2['TT2 Given/Denied Status'] == 'pichli bear diya gya') | (df2['TT2 Given/Denied Status'] == 'booster diya\r\n ')|(df2['TT2 Given/Denied Status'] == 'booster diya') | (df2['TT2 Given/Denied Status'] == 'buster dose')|(df2['TT2 Given/Denied Status'] == 'boostar') | (df2['TT2 Given/Denied Status'] == 'ttbusterdiya')), 'TT2 Given/Denied Status'] = 'Given'
df2['TT2 Given/Denied Status'].fillna('Unknown', inplace= True)
df2['TT1 Given/Denied Status'].fillna('Unknown', inplace= True)

# Extract the day, moth and year from ANC1 and ANC2 dates. ANC3 and 4 are not considered as they contain very little real values
for i in ['ANC1 Date', 'ANC2 Date']:
    df2[i[:4] + 'Year'] = df2[i].map(lambda x: pd.to_datetime(x).date().strftime('%Y').astype(int) if x != '-1' else -1)
    df2[i[:4] + 'Month'] = df2[i].map(lambda x: pd.to_datetime(x).date().strftime('%m').astype(int) if x != '-1' else -1)
    df2[i[:4] + 'Day'] = df2[i].map(lambda x: pd.to_datetime(x).date().strftime('%d').astype(int) if x != '-1' else -1)

# Drop the columns after extracting data
df2.drop(['ANC1 Date', 'ANC3 Date', 'ANC2 Date', 'ANC4 Date'], axis = 1, inplace = True)

# Take a look at the unique values in each column to see whether binning and such is required
for i in df2.columns:
    print (i, df2[i].unique() , '\n')   
# Note: Here, in the 'Villages' column, out of unique villages identified, many appear to be the same, but shown as uniqe,
# because of minor variations in spelling. However, at the moment, I am not venturing into the Herculean task of attempting to correct those
# variations, although it may lead to improved performance from the classifier.

# There are values like 35,36 and 40+ in pregnancy, which is clearly bad data, so set those values to -1
# Also more than 8 pregnancies below he age of 25 is physically impossible, so set it to -1
df2.loc[(df2['Total Pregnancies'] > 8) & (df2['Age'] < 25), 'Total Pregnancies'] = -1
df2.loc[(df2['Total Pregnancies'] >= 10) & (df2['Age'] < 28), 'Total Pregnancies'] = -1
df2.loc[(df2['Total Pregnancies'] >= 12) , 'Total Pregnancies'] = -1

# Since there is large variation in Heart Rate values, bin it to 10 groups nad lable in from 1-10
for i in ['ANC2 Fetal Heart Rate', 'ANC3 Fetal Heart Rate', 'ANC4 Fetal Heart Rate']:
    df2[i] = pd.cut(df2[i], 10, labels = [j for j in range(1,11)])

# Another instance of bad data, showing value in below column to be 17600, set to 0    
df2.loc[df2['FA Tablets given before 12 weeks (number)'] == 17600, 'FA Tablets given before 12 weeks (number)'] = 0

# Bin the follwing columns
df2['FA Tablets given before 12 weeks (number)'] = pd.cut(df2['FA Tablets given before 12 weeks (number)'], 7, labels = [1,2,3,4,5,6,7])
df2['IFA Tablets given after 12 weeks (number)'] = pd.cut(df2['IFA Tablets given after 12 weeks (number)'], 10, labels = [i for i in range(1,11)])
df2['No. of characters over 4 ANC\'s risk symptoms'] = pd.cut(df2['No. of characters over 4 ANC\'s risk symptoms'], 7, labels= [i for i in range(1,8)])

# Encode the string values in corresponding columns
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
obj_cols = [i for i in df2.columns if df2[i].dtype == 'O']
for i in obj_cols:
    df2[i] = lb.fit_transform(df2[i])
    
# import necessary items for training and prediction
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score

y = df2.Outcome
X = df2.drop('Outcome', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 500, max_features = 35, max_depth = 12, random_state = 0,  class_weight = \
        ({0:1000, 1:1000, 2:0.01, 3:1000, 4:1000}), ).fit(X_train, y_train)

cross_val_score(rf, X, y, n_jobs = -1, verbose = 5, cv = 5).mean()
