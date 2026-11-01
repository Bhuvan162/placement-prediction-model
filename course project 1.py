import pandas as pd
df=pd.read_csv('C:/Users/bhuva/idle projects/student_placement_500_analysis.csv')
print(f"Shape:{df.shape}\n")
print(f"DataTypes:\n{df.dtypes}\n")
print(f"Missing Values:\n{df.isnull().sum()}\n")#for missing values
print(f"Unique entries of 'Department':\n{df['Department'].unique()}\n")# for inconsistencies
print(f"Unique Entries of 'Placed':\n{df['Placed'].unique()}\n")#for inconsistencies

#to check duplicates
print(f"Exact Duplicates:{df[df.duplicated(keep=False)]}\n")
# so we will remove all the exact duplicates keeping the first one
df=df.drop_duplicates()

#for grouping purpose
'''print(df[df['Department']=='CSE'],
      df[df['Department']=='ECE'],
      df[df['Department']=='EEE'],
      df[df['Department']=='IT'],
      df[df['Department']=='AIDS'],
      df[df['Department']=='MECH'],)'''

#since the data is skewed we will use median imputation
for s in ['CGPA','Attendance','Internships','Technical_Score','Communication_Score']:
    
    df[s]=df[s].fillna(df[s].median())


#to check the outliers of the data we will use IQR method
for s in ['CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score']: 
    Q1=df[s].quantile(0.25)
    Q3=df[s].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=df[s][(df[s]<lower_bound)|(df[s]>upper_bound)]
    print(f"The number of outliers in {s} is :\n {outliers}",end='\n\n')


#for outliers we will cap them in lower and upper bound
'''for s in ['CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score']:

    df[s]=df[s].clip(lower=lower_bound,upper=upper_bound)'''
#since the outliers may have the chance to be genuine ,we will not remove the outliers 


#we get cleaned data ,now encoding   
df['Placed']=df['Placed'].map({'Yes':1,'No':0})
#df['Department']=df['Department'].map({'CSE':0,'ECE':1,'IT':2,'EEE':3,'AIDS':4,'MECH':5})
#since doing ordinal encoding may disrupt the correlation so we will first group by the data
#into sub datas of different departments and then train the model


#so to divide the data into subdata
dk1=df[df['Department']=='CSE']#cse 125 rows
dk2=df[df['Department']=='ECE']#83 rows
dk3=df[df['Department']=='EEE']#55 rows
dk4=df[df['Department']=='IT']#81 rows
dk5=df[df['Department']=='AIDS']#87 rows
dk6=df[df['Department']=='MECH']#69 rows




#now to remove uneseccary columns for prediction

'''for s in ['Department','CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score','Placed']: 
    dk[s]=df[s]'''
dp1=dk1[['CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score','Placed']]
dp2=dk2[['CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score','Placed']]
dp3=dk3[['CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score','Placed']]
dp4=dk4[['CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score','Placed']]
dp5=dk5[['CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score','Placed']]
dp6=dk6[['CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score','Placed']]
info={'CSE':dp1,'ECE':dp2,'EEE':dp3,'IT':dp4,'AIDS':dp5,'MECH':dp6}

#now we will do correlational matrix
for s in list(info.keys()):
    print(f"Correlation matrix of {s} Department:\n{info[s].corr()}",end='\n\n')
