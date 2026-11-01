import pandas as pd
df=pd.read_csv('C:/Users/bhuva/idle projects/student_placement_500_analysis.csv')
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())#for missing values
print(df['Department'].unique())# for inconsistencies
print(df['Placed'].unique())#for inconsistencies

#to check duplicates
print(df[df.duplicated(keep=False)])
# so we will remove all the exact duplicates keeping the first one
df=df.drop_duplicates()

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
    print(f"The number of outliers in {s} is :\n {outliers}")


#for outliers we will cap them in lower and upper bound
for s in ['CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score']:

    df[s]=df[s].clip(lower=lower_bound,upper=upper_bound)



#we get cleaned data ,now encoding   
df['Placed']=df['Placed'].map({'Yes':1,'No':0})
df['Department']=df['Department'].map({'CSE':0,'ECE':1,'IT':2,'EEE':3,'AIDS':4,'MECH':6})



#now to remove uneseccary columns for prediction

'''for s in ['Department','CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score','Placed']: 
    dk[s]=df[s]'''
dk=df[['Department','CGPA','Attendance','Backlogs','Internships','Projects','Aptitude_Score','Technical_Score','Communication_Score','Placed']]


#now we will do correlational matrix
print(dk.corr())
