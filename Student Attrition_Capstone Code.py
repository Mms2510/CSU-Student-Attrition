# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 11:26:10 2019

@author: EXT_BCR
"""

import numpy as np
import pandas as pd
import os

os.getcwd()

os.chdir("C:\\Users\\ext_bcr\\Desktop\\Jigsaw\\Dec 15 - Capstone Project")
data11 = pd.read_excel('Student Applications & Performance - Copy.xlsx')

data11.head(13)
data11.shape
data11.dtypes

data11.count()
data11.describe()

# reversing the o and 1s in Dependent variable

data11['STUDENT_LEFT'] = data11.RETURNED_2ND_YR
data11.STUDENT_LEFT.value_counts()
data11.STUDENT_LEFT.replace(0,2,inplace=True)
data11.STUDENT_LEFT.replace(1,0,inplace=True)
data11.STUDENT_LEFT.replace(2,1,inplace=True)
data11.STUDENT_LEFT.value_counts()
data11.RETURNED_2ND_YR.value_counts()

data11.STDNT_BACKGROUND.replace({'BGD 1':'BGD_1','BGD 2':'BGD_2','BGD 3':'BGD_3','BGD 4':'BGD_4','BGD 5':'BGD_5','BGD 6':'BGD_6','BGD 7':'BGD_7','BGD 8':'BGD_8'},inplace=True)



r = data11.STDNT_TEST_ENTRANCE_COMB.mode()
data11.STDNT_TEST_ENTRANCE_COMB.value_counts(dropna=False)

data11['STDNT_TEST_ENTRANCE_COMB'] = data11.STDNT_TEST_ENTRANCE_COMB.fillna(950)

data11.STDNT_TEST_ENTRANCE_COMB.isnull().sum()

data11.FIRST_TERM.head(10)
data11.FIRST_TERM.value_counts()


data11.DISTANCE_FROM_HOME = data11.DISTANCE_FROM_HOME.fillna(69)

data11.HIGH_SCHL_GPA = data11.HIGH_SCHL_GPA.fillna(data11.HIGH_SCHL_GPA.mean())

data11.count()

data11.SECOND_TERM_ATTEMPT_HRS.value_counts(dropna=False)

data11['SEC_TERM_PERFORMANCE'] = data11.SECOND_TERM_EARNED_HRS/data11.SECOND_TERM_ATTEMPT_HRS

data11.SECOND_TERM_EARNED_HRS.value_counts(dropna=False)

data11.count()

data11.SEC_TERM_PERFORMANCE.describe()
data11.SEC_TERM_PERFORMANCE.quantile(np.arange(0.95,1,0.001))

data11.SEC_TERM_PERFORMANCE[data11.SEC_TERM_PERFORMANCE >1]
data11.SEC_TERM_PERFORMANCE = data11.SEC_TERM_PERFORMANCE.replace(data11.SEC_TERM_PERFORMANCE[data11.SEC_TERM_PERFORMANCE >1],1)

data11.SEC_TERM_PERFORMANCE = data11.SEC_TERM_PERFORMANCE.fillna(0) # as no attempt and no points earned

data11['FIRST_TERM_PERFORMANCE'] = data11.FIRST_TERM_EARNED_HRS/data11.FIRST_TERM_ATTEMPT_HRS

data11['FIRST_TERM_PERFORMANCE'].describe()

data11.FIRST_TERM_PERFORMANCE[data11.FIRST_TERM_PERFORMANCE >1]
data11.FIRST_TERM_PERFORMANCE = data11.FIRST_TERM_PERFORMANCE.replace(data11.FIRST_TERM_PERFORMANCE[data11.FIRST_TERM_PERFORMANCE >1],1)

data11.CORE_COURSE_NAME_1_F.head(10)

data11.CORE_COURSE_NAME_1_F.str.split()

data11[['FIRSTSEM_COURSENAME_1','uselessnumbers']] = data11.CORE_COURSE_NAME_1_F.str.split(" ",expand=True)

data11.FIRSTSEM_COURSENAME_1.head()
data11.uselessnumbers.head()

data11[['FIRSTSEM_COURSENAME_2','uselessnumbers']] = data11.CORE_COURSE_NAME_2_F.str.split(" ",expand=True)
data11[['FIRSTSEM_COURSENAME_3','uselessnumbers']] = data11.CORE_COURSE_NAME_3_F.str.split(" ",expand=True)
data11[['FIRSTSEM_COURSENAME_4','uselessnumbers']] = data11.CORE_COURSE_NAME_4_F.str.split(" ",expand=True)

# Life and Career Planning in 5 Course Name
LACP =  data11.query("CORE_COURSE_NAME_5_F == 'Life and Career Planning'")['CORE_COURSE_NAME_5_F']
data11.CORE_COURSE_NAME_5_F = data11.CORE_COURSE_NAME_5_F.replace(LACP,'LACP 0070')
data11[['FIRSTSEM_COURSENAME_5','uselessnumbers']] = data11.CORE_COURSE_NAME_5_F.str.split(" ",expand=True)

# U.S. History to 1865 vin 6 Course name
USHT =  data11.query("CORE_COURSE_NAME_6_F == 'U.S. History to 1865'")['CORE_COURSE_NAME_6_F']
data11.CORE_COURSE_NAME_6_F = data11.CORE_COURSE_NAME_6_F.replace(USHT,'USHT 0070')
data11[['FIRSTSEM_COURSENAME_6','uselessnumbers']] = data11.CORE_COURSE_NAME_6_F.str.split(" ",expand=True)

#Second Sem Name

data11[['SECONDSEM_1_COURSENAME','uselessnumbers']] = data11.CORE_COURSE_NAME_1_S.str.split(" ",expand=True)
data11[['SECONDSEM_2_COURSENAME','uselessnumbers']] = data11.CORE_COURSE_NAME_2_S.str.split(" ",expand=True)
data11[['SECONDSEM_3_COURSENAME','uselessnumbers']] = data11.CORE_COURSE_NAME_3_S.str.split(" ",expand=True)
data11[['SECONDSEM_4_COURSENAME','uselessnumbers']] = data11.CORE_COURSE_NAME_4_S.str.split(" ",expand=True)
data11[['SECONDSEM_5_COURSENAME','uselessnumbers']] = data11.CORE_COURSE_NAME_5_S.str.split(" ",expand=True)
data11[['SECONDSEM_6_COURSENAME','uselessnumbers']] = data11.CORE_COURSE_NAME_6_S.str.split(" ",expand=True)

data11.count()
firstsemname = data11[['FIRSTSEM_COURSENAME_1','FIRSTSEM_COURSENAME_2','FIRSTSEM_COURSENAME_3','FIRSTSEM_COURSENAME_4','FIRSTSEM_COURSENAME_5','FIRSTSEM_COURSENAME_6']]
secondsemname = data11[['SECONDSEM_1_COURSENAME','SECONDSEM_2_COURSENAME','SECONDSEM_3_COURSENAME','SECONDSEM_4_COURSENAME','SECONDSEM_5_COURSENAME','SECONDSEM_6_COURSENAME']]

data11.count()
data11[['FIRSTSEM_COURSENAME_1','FIRSTSEM_COURSENAME_2','FIRSTSEM_COURSENAME_3', \
	'FIRSTSEM_COURSENAME_4','FIRSTSEM_COURSENAME_5','FIRSTSEM_COURSENAME_6']] \
	= data11[['FIRSTSEM_COURSENAME_1','FIRSTSEM_COURSENAME_2','FIRSTSEM_COURSENAME_3',\
	   'FIRSTSEM_COURSENAME_4','FIRSTSEM_COURSENAME_5','FIRSTSEM_COURSENAME_6']].fillna('NotEnrolled')
data11[['SECONDSEM_1_COURSENAME','SECONDSEM_2_COURSENAME','SECONDSEM_3_COURSENAME',\
	'SECONDSEM_4_COURSENAME','SECONDSEM_5_COURSENAME','SECONDSEM_6_COURSENAME']] \
	= data11[['SECONDSEM_1_COURSENAME','SECONDSEM_2_COURSENAME','SECONDSEM_3_COURSENAME',\
	   'SECONDSEM_4_COURSENAME','SECONDSEM_5_COURSENAME','SECONDSEM_6_COURSENAME']].fillna('NotEnrolled')


data11[['SECONDSEM_1_COURSENAME','SECONDSEM_2_COURSENAME','SECONDSEM_3_COURSENAME', \
	'SECONDSEM_4_COURSENAME','SECONDSEM_5_COURSENAME','SECONDSEM_6_COURSENAME']].head()

data11[['CORE_COURSE_GRADE_1_F','CORE_COURSE_GRADE_2_F','CORE_COURSE_GRADE_3_F','CORE_COURSE_GRADE_4_F','CORE_COURSE_GRADE_5_F','CORE_COURSE_GRADE_6_F']] =data11[['CORE_COURSE_GRADE_1_F','CORE_COURSE_GRADE_2_F','CORE_COURSE_GRADE_3_F','CORE_COURSE_GRADE_4_F','CORE_COURSE_GRADE_5_F','CORE_COURSE_GRADE_6_F']].fillna('NoGrades_F')

data11[['CORE_COURSE_GRADE_1_S','CORE_COURSE_GRADE_2_S','CORE_COURSE_GRADE_3_S','CORE_COURSE_GRADE_4_S','CORE_COURSE_GRADE_5_S','CORE_COURSE_GRADE_6_S']] = data11[['CORE_COURSE_GRADE_1_S','CORE_COURSE_GRADE_2_S','CORE_COURSE_GRADE_3_S','CORE_COURSE_GRADE_4_S','CORE_COURSE_GRADE_5_S','CORE_COURSE_GRADE_6_S']].fillna('NoGrades_S')
data11.CORE_COURSE_GRADE_5_S.head()


data11[['UNMET_NEED','GROSS_FIN_NEED','COST_OF_ATTEND', 'EST_FAM_CONTRIBUTION',]].head()
data11['FIN_AID'] = data11.GROSS_FIN_NEED - data11.UNMET_NEED
data11['FIN_AID'].head()

data.UNMET_NEED.describe()

lessthan0 = data11.UNMET_NEED[data11.UNMET_NEED<0]
data11.UNMET_NEED = data11.UNMET_NEED.replace(lessthan0,0)

data11.dtypes
data11.columns
data11.count()


rfdf111 = data11[['STDNT_AGE', 'STDNT_GENDER', 'STDNT_BACKGROUND',
       'IN_STATE_FLAG', 'INTERNATIONAL_STS', 'STDNT_MAJOR', 'STDNT_MINOR',
       'STDNT_TEST_ENTRANCE_COMB', 'FIRST_TERM','STUDENT_LEFT',
       'CORE_COURSE_GRADE_1_F','CORE_COURSE_GRADE_2_F',
       'CORE_COURSE_GRADE_3_F','CORE_COURSE_GRADE_4_F',
       'CORE_COURSE_GRADE_5_F','CORE_COURSE_GRADE_6_F', 'SECOND_TERM',
       'CORE_COURSE_GRADE_1_S','CORE_COURSE_GRADE_2_S',
       'CORE_COURSE_GRADE_3_S','CORE_COURSE_GRADE_4_S',
       'CORE_COURSE_GRADE_5_S','CORE_COURSE_GRADE_6_S', 'HOUSING_STS',
       'FATHER_HI_EDU_DESC',
       'MOTHER_HI_EDU_DESC','DEGREE_GROUP_DESC',
       'GROSS_FIN_NEED', 'UNMET_NEED','FIN_AID',
       'SEC_TERM_PERFORMANCE', 'FIRST_TERM_PERFORMANCE',
       'FIRSTSEM_COURSENAME_1','FIRSTSEM_COURSENAME_2',
       'FIRSTSEM_COURSENAME_3', 'FIRSTSEM_COURSENAME_4',
       'FIRSTSEM_COURSENAME_5', 'FIRSTSEM_COURSENAME_6',
       'SECONDSEM_1_COURSENAME', 'SECONDSEM_2_COURSENAME',
       'SECONDSEM_3_COURSENAME', 'SECONDSEM_4_COURSENAME',
       'SECONDSEM_5_COURSENAME', 'SECONDSEM_6_COURSENAME',
       'FIRSTSEM_NO_OF_COURSEOPT','SECONDSEM_NO_OF_COURSESOPT']]



rfdf111.head()
rfdf111.count()
rfdf111.shape
rfdf111.dtypes

rfdf111.SECONDSEM_1_COURSENAME.value_counts()

X1 = rfdf111.drop('STUDENT_LEFT', axis=1)
X1.shape
y1=rfdf111.STUDENT_LEFT
y1.head()

X1 = pd.get_dummies(X1)
X1.shape


#RF package install

from sklearn.model_selection import train_test_split
X_train1,X_test1,y_train1,y_test1 = train_test_split(X1,y1,test_size = 0.30, random_state=100)
X_train1.shape


from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(n_estimators=100, oob_score=True,n_jobs=-1,random_state=200)
clf1.fit(X_train1,y_train1)

clf1.oob_score_

for w in range(10,300,20):
    clf1=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=200)
    clf1.fit(X_train1,y_train1)
    oob1=clf1.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob1))
    print('************************')

for w in range(250,270,2):
    clf1=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=200)
    clf1.fit(X_train1,y_train1)
    oob1=clf1.oob_score_
    print('For n_estimators = '+str(w))
    print('OOB score is '+str(oob1))
    print('************************')
    
   # Best score coming in 258
# For n_estimators = 258
#OOB score is 0.8306722689075631
   

# 258 Gave the best oob Scroe --- Finalizing
clf1=RandomForestClassifier(n_estimators=258,oob_score=True,n_jobs=-1,random_state=200)
clf1.fit(X_train1,y_train1)

clf1.oob_score_

clf1.feature_importances_

imp_feature1 = pd.Series(clf1.feature_importances_, index=X1.columns.tolist())

impdf = imp_feature1.sort_values(ascending=False)
impdf.head(30)

impdf.head(30).plot(kind='bar')

#Logistic
data11['TOTAL_NO_COURSES'] = data11.SECONDSEM_NO_OF_COURSESOPT + data11.FIRSTSEM_NO_OF_COURSEOPT



logdfdummy = data11[['STUDENT IDENTIFIER','STDNT_AGE', 'STDNT_GENDER', 'STDNT_BACKGROUND',
       'IN_STATE_FLAG', 'INTERNATIONAL_STS', 'STDNT_MAJOR', 'STDNT_MINOR',
       'STDNT_TEST_ENTRANCE_COMB', 'FIRST_TERM','STUDENT_LEFT',
       'CORE_COURSE_GRADE_1_F','CORE_COURSE_GRADE_2_F',
       'CORE_COURSE_GRADE_3_F','CORE_COURSE_GRADE_4_F',
       'CORE_COURSE_GRADE_5_F','CORE_COURSE_GRADE_6_F', 'SECOND_TERM',
       'CORE_COURSE_GRADE_1_S','CORE_COURSE_GRADE_2_S',
       'CORE_COURSE_GRADE_3_S','CORE_COURSE_GRADE_4_S',
       'CORE_COURSE_GRADE_5_S','CORE_COURSE_GRADE_6_S', 'HOUSING_STS',
       'DISTANCE_FROM_HOME', 'HIGH_SCHL_GPA','FATHER_HI_EDU_DESC',
       'MOTHER_HI_EDU_DESC','DEGREE_GROUP_DESC',
       'GROSS_FIN_NEED', 'UNMET_NEED','FIN_AID',
       'SEC_TERM_PERFORMANCE', 'FIRST_TERM_PERFORMANCE',
       'FIRSTSEM_COURSENAME_1','FIRSTSEM_COURSENAME_2',
       'FIRSTSEM_COURSENAME_3', 'FIRSTSEM_COURSENAME_4',
       'FIRSTSEM_COURSENAME_5', 'FIRSTSEM_COURSENAME_6',
       'SECONDSEM_1_COURSENAME', 'SECONDSEM_2_COURSENAME',
       'SECONDSEM_3_COURSENAME', 'SECONDSEM_4_COURSENAME',
       'SECONDSEM_5_COURSENAME', 'SECONDSEM_6_COURSENAME',
       'FIRSTSEM_NO_OF_COURSEOPT','SECONDSEM_NO_OF_COURSESOPT']]

logdfdummy.shape


logdfdummy1 = pd.get_dummies(logdfdummy)
"""
logdata = logdfdummy[['STUDENT IDENTIFIER','STUDENT_LEFT','SEC_TERM_PERFORMANCE','CORE_COURSE_GRADE_1_S_NoGrades_S','HIGH_SCHL_GPA',
	      'SECONDSEM_NO_OF_COURSESOPT','STDNT_TEST_ENTRANCE_COMB','DISTANCE_FROM_HOME',
	      'FIRST_TERM_PERFORMANCE','SECONDSEM_2_COURSENAME_NotEnrolled',
	     'SECONDSEM_1_COURSENAME_NotEnrolled','SECOND_TERM',
	     'FIRST_TERM','CORE_COURSE_GRADE_2_S_NoGrades_S',
	     'GROSS_FIN_NEED','UNMET_NEED','FIRSTSEM_NO_OF_COURSEOPT',
	     'CORE_COURSE_GRADE_1_S_B','STDNT_AGE','HOUSING_STS_Off Campus',
	     'CORE_COURSE_GRADE_1_S_F','MOTHER_HI_EDU_DESC_High School',
	     'CORE_COURSE_GRADE_1_S_C','MOTHER_HI_EDU_DESC_College/Beyond',
	     'SECONDSEM_1_COURSENAME_ENGL','FATHER_HI_EDU_DESC_High School',
	     'CORE_COURSE_GRADE_2_S_F','FATHER_HI_EDU_DESC_College/Beyond',
	     'STDNT_BACKGROUND_BGD 1','STDNT_GENDER_F',
	     'CORE_COURSE_GRADE_2_S_B','HOUSING_STS_On Campus']]
"""


#logdata['UNMET_NEEDX']=logdata.UNMET_NEED.map(lambda x:'True' if x >1 else 'False')
#logdata.UNMET_NEEDX.value_counts()

#logdata['GROSS_FIN_NEEDX']=logdata.GROSS_FIN_NEED.map(lambda x:'Required' if x >0 else 'NotRequired')
#logdata.GROSS_FIN_NEEDX.value_counts()

#Give only one at a time
#logdf = logdata.drop(['UNMET_NEED','GROSS_FIN_NEED'],axis=1)

logdf = logdfdummy1



logdf.HIGH_SCHL_GPA.describe()
logdf.HIGH_SCHL_GPA.quantile(np.arange(0,0.01,0.001))

logdf.HIGH_SCHL_GPA = logdf.HIGH_SCHL_GPA.replace(logdf.HIGH_SCHL_GPA[logdf.HIGH_SCHL_GPA == 0],1.87)


logdf.rename(columns={'FATHER_HI_EDU_DESC_High School':'FATHER_HI_EDU_DESC_High_School','HOUSING_STS_On Campus':'HOUSING_STS_On_Campus', \
			'STDNT_BACKGROUND_BGD 1':'STDNT_BACKGROUND_BGD_1','HOUSING_STS_Off Campus':'HOUSING_STS_Off_Campus', \
			'MOTHER_HI_EDU_DESC_High School':'MOTHER_HI_EDU_DESC_High_School'},inplace=True)
logdf.rename(columns={'MOTHER_HI_EDU_DESC_College/Beyond':'MOTHER_HI_EDU_DESC_CollegeBeyond', \
		      'FATHER_HI_EDU_DESC_College/Beyond':'FATHER_HI_EDU_DESC_CollegeBeyond'},inplace=True)

logdf1= logdf


logdf1['TOTAL_NO_COURSES'] = logdf1.SECONDSEM_NO_OF_COURSESOPT + logdf1.FIRSTSEM_NO_OF_COURSEOPT


logdf_train = logdf1.sample(frac =0.70 , random_state=150)
logdf_test = logdf1.drop(logdf_train.index)

logdf_train.shape
logdf_test.shape

logdf_train.FIN_AID.head()

# LOG Model start
import statsmodels.formula.api as smf
import statsmodels.api as sm

logdf_train.count()
logdf.count()



def mod(yX,number):
	model11=smf.glm(yX,data = logdf_train, family=sm.families.Binomial()).fit()
	print('LogisticModel',number)
	print(model11.summary())
	return model11



yX = "STUDENT_LEFT~SEC_TERM_PERFORMANCE+ \
	CORE_COURSE_GRADE_2_S_NoGrades_S+SECONDSEM_2_COURSENAME_NotEnrolled+ \
	 TOTAL_NO_COURSES+   \
	HOUSING_STS_On_Campus+  \
	 INTERNATIONAL_STS_Y+  \
	  \
	CORE_COURSE_GRADE_2_S_D+ \
	STDNT_BACKGROUND_BGD_1+ \
	STDNT_GENDER_F+STDNT_MAJOR_Psychology"
mod(yX,1)


# CHeck Multi Coolinearity

y=logdf_train.STUDENT_LEFT
X1= logdf_train.drop('STUDENT_LEFT',axis=1)
y.head()

from statsmodels.stats.outliers_influence import variance_inflation_factor


from patsy import dmatrices
y, X1 = dmatrices("STUDENT_LEFT~SEC_TERM_PERFORMANCE+ \
	CORE_COURSE_GRADE_2_S_NoGrades_S+SECONDSEM_2_COURSENAME_NotEnrolled+ \
	 TOTAL_NO_COURSES+   \
	HOUSING_STS_On_Campus+  \
	 INTERNATIONAL_STS_Y+  \
	  \
	CORE_COURSE_GRADE_2_S_D+ \
	STDNT_BACKGROUND_BGD_1+ \
	STDNT_GENDER_F+STDNT_MAJOR_Psychology",
		   data=logdf_train, return_type='dataframe')



vif=pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif
vif['Independent_Var'] = X1.columns
vif.round(1)
vif.round(2)
vif.round(3)

##Add that in model removing ---CORE_COURSE_GRADE_2_S_NoGrades_S
finalmodel=smf.glm("STUDENT_LEFT~SEC_TERM_PERFORMANCE+ \
	SECONDSEM_2_COURSENAME_NotEnrolled+ \
	 TOTAL_NO_COURSES+   \
	HOUSING_STS_On_Campus+  \
	 INTERNATIONAL_STS_Y+  \
	  \
	CORE_COURSE_GRADE_2_S_D+ \
	STDNT_BACKGROUND_BGD_1+ \
	STDNT_GENDER_F+STDNT_MAJOR_Psychology",data = logdf_train, family=sm.families.Binomial()).fit()
	
print(finalmodel.summary())
	
## Let's check confusion matrix and AUC
import sklearn.metrics as metrics
y_true=logdf_test['STUDENT_LEFT']
y_pred=finalmodel.predict(logdf_test)

y_pred.head()


y_true=logdf_test['STUDENT_LEFT']
y_pred=finalmodel.predict(logdf_test).map(lambda x:1 if x > 0.6 else 0)
metrics.confusion_matrix(y_true,y_pred)


y_score=finalmodel.predict(logdf_test)
fpr,tpr,thresholds=metrics.roc_curve(y_true,y_score)
x,y=np.arange(0,1.1,0.1),np.arange(0,1.1,0.1)


import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(fpr,tpr,"-")
plt.plot(x,y,'b--')

#Extracting the TPR and FPR for Tableau visualization

df = pd.DataFrame()
df['False+ve_Rate']=pd.Series(fpr)
df['True+ve_Rate']=pd.Series(tpr)

df.head()
df.to_csv(r'C:\\Users\\ext_bcr\\Desktop\\Jigsaw\\Dec 15 - Capstone Project\\TPR_FPR.csv')



## AUC
metrics.roc_auc_score(y_true,y_score)

#Gains

logdf_test['prob']=finalmodel.predict(logdf_test)
logdf_test['prob'].head()

logdf_test['prob_deciles']=pd.qcut(logdf_test['prob'],q=10)

logdf_test.sort_values('prob',ascending=False).head()


gains=logdf_test.groupby("prob_deciles",as_index=False)['STUDENT_LEFT'].agg(['sum','count']).reset_index().sort_values("prob_deciles",
                 ascending=False)

gains.head(10)

gains.columns=["Deciles","TotalEvents","NumberObs"]

gains["PercEvents"]=gains['NumberObs']/gains['TotalEvents'].sum()
gains["CumulativeEvents"]=gains.PercEvents.cumsum()
gains

## These are the STUDENTS to target
logdf_test.sort_values("prob",ascending=False)[['STUDENT IDENTIFIER','prob']].head(90)

