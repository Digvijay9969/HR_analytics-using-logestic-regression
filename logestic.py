import os
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import figure
os.chdir("D:/D_S/HR_Analytics_project")

rawDf=pd.read_csv("train.csv")
rawDf.shape
rawDf.columns
predDf=pd.read_csv("test.csv")

#Adding "is_promoted" column in predDf
predDf['is_promoted']=0
predDf.columns

#sampling fullraw into train and test data
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawDf,train_size=0.8,random_state=97)

trainDf.shape
testDf.shape

#Adding Source column in trainDf,testDf,predDf
trainDf['Source']='Train'
testDf['Source']='Test'
predDf['Source']='Prediction'

#combine trainDf,testDf and predDf datasets
fullraw=pd.concat([trainDf,testDf,predDf],axis=0)
fullraw.columns

# split of 0s and 1s
fullraw.loc[fullraw['Source']=='Train','is_promoted'].value_counts()/fullraw[fullraw['Source']=='Train'].shape[0]

# 91.44 are not promoted and 8.56 are promoted

#summarize the data
fullraw_summary=fullraw.describe()

#removing identifire column
fullraw.drop(['employee_id'],axis=1,inplace=True)
fullraw.columns

#missing value check
fullraw.isna().sum()

#we have missing values in education and previous_year_rating
fullraw['education'].dtype #checking data type of column 

tempMode=fullraw.loc[fullraw['Source']=='Train','education'].mode()[0]
tempMode
#filling na values using mode
fullraw['education'].fillna(tempMode,inplace=True)

fullraw['previous_year_rating'].dtype
tempMode=fullraw.loc[fullraw['Source']=='Train','previous_year_rating'].mode()[0]
tempMode
#filling na values using mode
fullraw['previous_year_rating'].fillna(tempMode,inplace=True)

# =============================================================================
# use data description excel sheet to convert numeric variables to categorical variables 
# =============================================================================
#KPIs_met >80%,awards_won?,previous_year_rating
fullraw.columns

variableToUpdate='KPIs_met >80%'

#to check the uniwque categories of the variable
fullraw[variableToUpdate].value_counts()
fullraw[variableToUpdate].replace({0:'Bad',1:'Good'},inplace=True)
fullraw[variableToUpdate].value_counts()

variableToUpdate='awards_won?'

#to check the uniwque categories of the variable
fullraw[variableToUpdate].value_counts()
fullraw[variableToUpdate].replace({0:'No',1:'Yes'},inplace=True)
fullraw[variableToUpdate].value_counts()

variableToUpdate='previous_year_rating'

#to check the uniwque categories of the variable
fullraw[variableToUpdate].value_counts()
fullraw[variableToUpdate].replace({1:'Poor',2:'Bad',3:'Avg',4:'Good',5:'Best'},inplace=True)
fullraw[variableToUpdate].value_counts()

# =============================================================================
# Bivariate analysis using box plot
# =============================================================================
import seaborn as sns
trainDf=fullraw.loc[fullraw['Source']=='Train']
continuousVars=trainDf.columns[trainDf.dtypes != object]
continuousVars

fileName="continuous_bivariate_analysis.pdf"
pdf=PdfPages(fileName)
for colNumber,colName in enumerate(continuousVars):
    figure()
    sns.boxplot(y=trainDf[colName],x=trainDf["is_promoted"])
    pdf.savefig(colNumber+1)
pdf.close()

# =============================================================================
# Bivariate analysis using histogram 
# =============================================================================
categoricalVars = trainDf.columns[trainDf.dtypes== object]
categoricalVars

# sns.histplot(trainDf,x='gender',hue='is_promoted',stat='probability',multiple='fill')

fileName= "Categorical_Bivariate_Analysis_hist.pdf"
pdf=PdfPages(fileName)
for colNumber,colName in enumerate(categoricalVars):
    figure()
    sns.histplot(trainDf,x=colName,hue='is_promoted',stat='probability',multiple='fill')
    pdf.savefig(colNumber+1)
pdf.close()

# =============================================================================
# Dummy variable creation
# =============================================================================
#souce column will change to source_train and it contains 0s and 1s
fullraw2=pd.get_dummies(fullraw,drop_first = True)
fullraw2.shape


# =============================================================================
# Divide the data into Train and Test
# =============================================================================

# divide the data into train and test based on source and column and make sure you drop the source column

#step 1: Divide into Train and Test
Train = fullraw2[fullraw2['Source_Train']==1].drop(['Source_Train','Source_Test'],axis=1).copy()
Train.shape

Test=fullraw2[fullraw2['Source_Test']==1].drop(['Source_Test','Source_Train'],axis=1)
Test.shape

Prediction=fullraw2[(fullraw2['Source_Train']==0) & 
                    (fullraw2['Source_Test']==0)].drop(['Source_Train','Source_Test'],axis=1)
Prediction.shape

#step2: Divide independent and dependent columns
depVar = "is_promoted"

trainX=Train.drop([depVar],axis=1).copy()
trainX.shape

trainY=Train[depVar].copy()
trainY.shape

testX=Test.drop([depVar],axis = 1).copy()
testY=Test[depVar].copy()

predX=Prediction.drop([depVar],axis=1).copy()

# =============================================================================
# Add Intercept column
# =============================================================================

from statsmodels.api import add_constant
trainX=add_constant(trainX)
testX=add_constant(testX)
predX=add_constant(predX)

# =============================================================================
# VIF check
# =============================================================================

from statsmodels.stats.outliers_influence import variance_inflation_factor
tempMaxVIF= 10
maxVIF=10
trainXcopy=trainX.copy()
counter=1
highVIFColumnNames=[]

while(tempMaxVIF >= maxVIF):
    print(counter)
    
    #create an empty temporary df to store VIF values
    tempVIFDf=pd.DataFrame()
   
    #calculate vif using list comprehsion
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXcopy.values, i)for i in range(trainXcopy.shape[1])]
    
    #creating newcolumn 'column_name' to store the col names against the vif values form list compherension
    tempVIFDf['column_name']=trainXcopy.columns
    
    #Drop Na rows from the df if there is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    #Sort the df based on vif values, the pick the top most column name(which has the highest vif)
    tempColumnName=tempVIFDf.sort_values(['VIF'],ascending=False).iloc[0,1]
    
    #store the max vif value in tempMaxVIF
    tempMaxVIF=tempVIFDf.sort_values(['VIF'],ascending=False).iloc[0,0]
    
    if(tempMaxVIF >= maxVIF):
        trainXcopy=trainXcopy.drop(tempColumnName,axis=1)
        highVIFColumnNames.append(tempColumnName)
        print(tempColumnName)
    counter= counter+1

highVIFColumnNames
#we need to exclude const column getting removed or drop.This is intercept
highVIFColumnNames
trainX=trainX.drop(highVIFColumnNames,axis=1)
trainX.shape
testX=testX.drop(highVIFColumnNames,axis=1)
testX.shape
predX=predX.drop(highVIFColumnNames,axis=1)
predX.shape

# =============================================================================
# model building
# =============================================================================

#build logestic regression model using statsmodel 
from statsmodels.api import Logit
M1= Logit(trainY,trainX) #(dep_var,indep_var) # this is model defination
M1_model=M1.fit()
M1_model.summary()

# =============================================================================
# manual model selection. drop most insignificant variables
# =============================================================================

colsToDrop=['region_region_18']
M2=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M2.summary()

#Drop education_Below Secondary
colsToDrop.append('education_Below Secondary')
M3=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M3.summary()

#length_of_service
colsToDrop.append('length_of_service')
M4=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M4.summary()

#drop region_region_30
colsToDrop.append('region_region_30')
M5=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M5.summary()

#drop region_region_13
colsToDrop.append('region_region_13')
M6=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M6.summary()

# drop region_region_2
colsToDrop.append('region_region_2')
M7=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M7.summary()

#drop recruitment_channel_sourcing
colsToDrop.append('recruitment_channel_sourcing')
M8=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M8.summary()

# drop region_region_14
colsToDrop.append('region_region_14')
M9=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M9.summary()

# drop region_region_12
colsToDrop.append('region_region_12')
M10=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M10.summary()

# drop region_region_27
colsToDrop.append('region_region_27')
M11=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M11.summary()

# drop gender_m 
colsToDrop.append('gender_m')
M12=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M12.summary()

# drop region_region_15
colsToDrop.append('region_region_15')
M13=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M13.summary()

#department_Sales & Marketing 
colsToDrop.append('department_Sales & Marketing')
M14=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M14.summary()

#drop region_region_8
colsToDrop.append('region_region_8')
M15=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M15.summary()

#drop recruitment_channel_referred
colsToDrop.append('recruitment_channel_referred')
M16=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M16.summary()

#drop region_region_10
colsToDrop.append('region_region_10')
M17=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M17.summary()

#drop region_region_24
colsToDrop.append('region_region_24')
M18=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M18.summary()

#drop region_region_16
colsToDrop.append('region_region_16')
M19=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M19.summary()

#drop department_Procurement 
colsToDrop.append('department_Procurement')
M20=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M20.summary()

#drop region_region_3
colsToDrop.append('region_region_3')
M21=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M21.summary()

#drop region_region_26
colsToDrop.append('region_region_26')
M22=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M22.summary()

#drop region_region_19
colsToDrop.append('region_region_19')
M23=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M23.summary()

# =============================================================================
# Prediction and validation
# =============================================================================
trainX=trainX.drop(colsToDrop,axis=1)
testX=testX.drop(colsToDrop,axis=1)
predX=predX.drop(colsToDrop,axis=1)

#storing probability prediction in testX
testX['prob']=M23.predict(testX)

#new column of prob is created in testX
testX.columns
testX.shape
testX['prob'][0:6]
testY[:6]

#classify 0 and 1 with the cutoff 0.5
testX['test_class']=np.where(testX['prob']>0.5,1,0)
testX['test_class']

#confusion matrix
confusion_mat=pd.crosstab(testX['test_class'],testY)
confusion_mat

#check acurracy of model
(sum(np.diagonal(confusion_mat))/testX.shape[0])*100 # 91.58%

# =============================================================================
# Pricision,recall,f1 score
# =============================================================================

from sklearn.metrics import classification_report
print(classification_report(testY,testX['test_class']))

#Precision: TP/(TP+FP) 
#Recall : TP/(TP+FN) 
#F1Score : 2*Precision*Recall/(Presion+Recall)
#Precision,Recall, F1 Score interpretation: Higher the better
#Precision
#Intuitive understanding : How many of our "Predicted" is_promoted is "actual" is_promoted

# =============================================================================
# ROC curve
# =============================================================================

from sklearn.metrics import roc_curve,auc

#predict on train data
train_prob=M23.predict(trainX)

#calculate FPR, TPR, Cutoff Thresholds
fpr,tpr,cutoff=roc_curve(trainY, train_prob)

# Cutoff Table Creation
Cutoff_Table= pd.DataFrame()
Cutoff_Table['FPR']=fpr
Cutoff_Table['TPR']=tpr
Cutoff_Table['Cutoff']=cutoff

#Plot Roc Curve
import seaborn as sns
sns.lineplot(Cutoff_Table['FPR'],Cutoff_Table['TPR'])

auc(fpr,tpr)

# =============================================================================
# Improve model output using new cutoff point
# =============================================================================

Cutoff_Table['DiffBetweenTPRFPR']=Cutoff_Table['TPR']-Cutoff_Table['FPR']  #max diff between tpr and fpr
Cutoff_Values=Cutoff_Table.sort_values(['DiffBetweenTPRFPR'],ascending=False).iloc[0,2]
Cutoff_Values  #0.07352771306595264

Cutoff_Table['Distance']=np.sqrt((1-Cutoff_Table['TPR'])**2 + (0-Cutoff_Table['FPR'])**2)
Cutoff_Values1=Cutoff_Table.sort_values(['Distance'],ascending=True).iloc[0,2]
Cutoff_Values1 #0.0777386718919916

cutoff_point= 	0.11122168267912055


#0.10733892656214553


cutoff_point


#values tstdata bsased on new cutoff(difference method)
testX["test_class1"]=np.where(testX['prob']>=Cutoff_Values,1,0)
testX["test_class1"][0:6]

#
confusion_mat2=pd.crosstab(testX['test_class1'],testY) # predicted,actual
confusion_mat2
#classification report
print(classification_report(testY,testX['test_class1']))

# =============================================================================
# Prediction on prediction dataset
# =============================================================================

predX['is_promoted']=M23.predict(predX)
predX['is_promoted']=np.where(predX['is_promoted']>Cutoff_Values,1,0)
logistic_output=pd.DataFrame()
logistic_output=pd.concat([predDf['employee_id'],predX['is_promoted']],axis=1)
logistic_output.to_csv('logistic_output.csv',index=False)






