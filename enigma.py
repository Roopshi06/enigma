#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all the necessary libraries
import  pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing 
from sklearn import linear_model
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import tree
from sklearn import naive_bayes
from sklearn import neighbors

#loading the dataset
data=pd.read_csv('dataset - dataset.csv')
data.head()


# In[ ]:





# In[2]:


#display the structure of the dataset
data.info()


# In[3]:


#finding the number of null values in each column
data.isnull().sum()


# In[4]:


#replacing the column names as per our convenience
df=data.rename(columns=({'patient number':'p_no','resting blood pressure':'rbp','cholestrol':'chol','fasting blood sugar':'fbs'                          ,'resting electrocardiographic':'recg','maximum heart rate achieved':'max_hrt','body mass index':'bmi'                         ,'Admit Date':'adm_dt','Discharge Date':'dis_dt','pulse rate':'pulse_rate','left anterior descending':'LAD','circumflex artery':'CA'                         ,'right coronary artery':'RCA','obtuse marginal':'OM','left posterior descending artery':'LPD'                         ,'left coronary artery':'LCA'}))


# In[5]:


#displaying the structure to make sure that our column names have been successfully changed
df.info()


# In[6]:


#convert all the values to uppercase
df["sex"]=df["sex"].str.upper()


# In[7]:


#find out all the unique values of column 'sex'
df['sex'].unique()


# In[8]:


#as all the data is in same foramt so there is no need of any change at this point


# In[9]:


#find out all the unique values of column 'chest pain' along with its frequency
df['chest pain'].value_counts()


# In[10]:


#as all the data is in same foramt so there is no need of any change at this point


# In[11]:


#find out all the unique values of column 'smoking' along with its frequency
df["smoking"].value_counts()


# In[12]:


#as the data was incosistent,we converted all the entries to upper case
df["smoking"]=df["smoking"].str.upper()


# In[13]:


#find out all the unique values of column 'smoking' along with its frequency
df["smoking"].value_counts()


# In[14]:


#as the data is still inconsistent,we made the required changes
df['smoking']=df['smoking'].replace({"N0":"NO",'.':'NO'})


# In[15]:


df["smoking"].value_counts()


# In[16]:


#now the data of column 'smoking' is consistent


# In[17]:


#find out all the unique values of column 'age' 
df['age'].unique()


# In[18]:


#as all the data is in same foramt so there is no need of any change at this point


# In[19]:


#find out all the unique values of column 'rbp' along with its frequency
df["rbp"].value_counts()


# In[20]:


#as the data is inconsistent,we made the required changes
df['rbp']=df['rbp'].replace({"ca":"NaN"})


# In[21]:


#find out all the unique values of column 'rbp' ato make sure that our changes are made
df['rbp'].unique()


# In[22]:


#find out all the unique values of column 'chol'
df['chol'].unique()


# In[23]:


#as all the data is in same foramt so there is no need of any change at this point


# In[24]:


#find out all the unique values of column 'fbs' along with its frequency
df["fbs"].value_counts()


# In[25]:


#as all the data is in same foramt so there is no need of any change at this point


# In[26]:


#find out all the unique values of column 'recg' along with its frequency
df["recg"].value_counts()


# In[27]:


#as all the data is in same foramt so there is no need of any change at this point


# In[28]:


#find out all the unique values of column 'max_hrt' along with its frequency
df["max_hrt"].value_counts()


# In[29]:


#as all the data is in same foramt so there is no need of any change at this point


# In[30]:


#find out all the unique values of column 'oldpeak' along with its frequency
df["oldpeak"].value_counts()


# In[31]:


#as the data is not in same format,we made the necessary changes


# In[32]:


df['oldpeak']=df['oldpeak'].replace({"\n":"1"})


# In[33]:


#find out all the unique values of column 'oldpeak' along with its frequency to make sure that the necessary changes are done
df["oldpeak"].value_counts()


# In[34]:


#find out all the unique values of column 'slope' 
df['slope'].unique()


# In[35]:


#as all the data is in same foramt so there is no need of any change at this point


# In[36]:


#find out all the unique values of column 'exang' along with its frequency
df["exang"].value_counts()


# In[37]:


#as all the data is in same foramt so there is no need of any change at this point


# In[38]:


#find out all the unique values of column 'bmi' 
df['bmi'].unique()


# In[39]:


#as all the data is in same foramt so there is no need of any change at this point


# In[40]:


#find out all the unique values of column 'smoking' along with its frequency
df['pulse_rate'].unique()


# In[41]:


#as all the data is in same foramt so there is no need of any change at this point


# In[42]:


#find out all the unique values of column 'LAD' along with its frequency
df['LAD'].value_counts()


# In[43]:


#as the data was inconsistent,we made the necessary changes
df['LAD']=df['LAD'].replace({"Oct-20":"10-20"})


# In[44]:


#find out all the unique values of column 'LAD' along with its frequency to make sure that the required changes are done
df['LAD'].value_counts()


# In[45]:


#find out all the unique values of column 'CA' along with its frequency
df['CA'].value_counts()


# In[46]:


#as the data was inconsistent,we made the necessary changes
df['CA']=df['CA'].replace({"Oct-20":"10-20"})


# In[47]:


#find out all the unique values of column 'CA' along with its frequency to make sure that the required changes are done
df['CA'].value_counts()


# In[48]:


#find out all the unique values of column 'RCA' along with its frequency
df['RCA'].value_counts()


# In[49]:


#as the data was inconsistent,we made the necessary changes
df['RCA']=df['RCA'].replace({"Oct-20":"10-20"})


# In[50]:


#find out all the unique values of column 'RCA' along with its frequency to make sure that the required changes are done
df['RCA'].value_counts()


# In[51]:


#find out all the unique values of column 'OM' along with its frequency
df['OM'].value_counts()


# In[52]:


#as the data was inconsistent,we made the necessary changes
df['OM']=df['OM'].replace({"Oct-20":"10-20","40":"40-50"})
#find out all the unique values of column 'OM' along with its frequency to make sure that the required changes are done
df['OM'].value_counts()


# In[53]:


#find out all the unique values of column 'LPD' along with its frequency
df['LPD'].value_counts()


# In[54]:


#as the data was inconsistent,we made the necessary changes
df['LPD']=df['LPD'].replace({"Oct-20":"10-20"})
#find out all the unique values of column 'LPD' along with its frequency to make sure that the required changes are done
df['LPD'].value_counts()


# In[55]:


#find out all the unique values of column 'LCA' along with its frequency
df['LCA'].value_counts()


# In[56]:


#as the data was inconsistent,we made the necessary changes
df['LCA']=df['LCA'].replace({"Oct-20":"10-20"})
#find out all the unique values of column 'LCA' along with its frequency to make sure that the required changes are done
df['LCA'].value_counts()


# In[57]:


#find out all the unique values of column 'LCA' along with its frequency
df['stent'].value_counts()


# In[58]:


#graphical representation of our data
sns.countplot(x='stent',data=df)


# In[59]:


#graphical representation to show the effect of 'sex' on the no of 'stents'
sns.countplot(x='sex',hue='stent',data=df)


# In[60]:


#graphical representation to show the effect of 'age' on the no of 'stents'
sns.countplot(x='age',hue='stent',data=df)


# In[61]:


#graphical representation to show the effect of 'chest pain' on the no of 'stents'
sns.countplot(x='chest pain',hue='stent',data=df)


# In[62]:


#graphical representation to show the effect of 'rbp' on the no of 'stents'
sns.countplot(x='rbp',hue='stent',data=df)


# In[63]:


#graphical representation to show the effect of 'chol' on the no of 'stents'
sns.countplot(x='chol',hue='stent',data=df)


# In[64]:


#graphical representation to show the effect of 'fbs' on the no of 'stents'
sns.countplot(x='fbs',hue='stent',data=df)


# In[65]:


#graphical representation to show the effect of 'recg' on the no of 'stents'
sns.countplot(x='recg',hue='stent',data=df)


# In[66]:


#graphical representation to show the effect of 'max_hrt' on the no of 'stents'
sns.countplot(x='max_hrt',hue='stent',data=df)


# In[67]:


#graphical representation to show the effect of 'oldpeak' on the no of 'stents'
sns.countplot(x='oldpeak',hue='stent',data=df)


# In[68]:


#graphical representation to show the effect of 'exang' on the no of 'stents'
sns.countplot(x='exang',hue='stent',data=df)


# In[69]:


#graphical representation to show the effect of 'bmi' on the no of 'stents'
sns.countplot(x='bmi',hue='stent',data=df)


# In[70]:


#graphical representation to show the effect of 'pulse_rate' on the no of 'stents'
sns.countplot(x='pulse_rate',hue='stent',data=df)


# In[71]:


#graphical representation to show the effect of 'smoking' on the no of 'stents'
sns.countplot(x='smoking',hue='stent',data=df)


# In[72]:


#graphical representation to show the effect of 'LAD' on the no of 'stents'
sns.countplot(x='LAD',hue='stent',data=df)


# In[73]:


#graphical representation to show the effect of 'CA' on the no of 'stents'
sns.countplot(x='CA',hue='stent',data=df)


# In[74]:


#graphical representation to show the effect of 'RCA' on the no of 'stents'
sns.countplot(x='RCA',hue='stent',data=df)


# In[75]:


#graphical representation to show the effect of 'OM' on the no of 'stents'
sns.countplot(x='OM',hue='stent',data=df)


# In[76]:


#graphical representation to show the effect of 'LPD' on the no of 'stents'
sns.countplot(x='LPD',hue='stent',data=df)


# In[77]:


#graphical representation to show the effect of 'LCA' on the no of 'stents'
sns.countplot(x='LCA',hue='stent',data=df)


# In[78]:


#removing the unnecessary columns that will not have any affect on the final outcome
lst=['p_no','Name','adm_dt','dis_dt']


# In[79]:


df.drop(lst,axis=1,inplace=True)


# In[80]:


#displaying the structure of the dataset to make sure that the columns have been successfully dropped
df.info()


# In[81]:


#now,we are going to convert the catogorical data into numeric values in order to further calculations feasible
#we are creating the dummy variables of column 'sex':0-M,1-F
sex_m=pd.get_dummies(df['sex'],drop_first=True)
sex_m


# In[82]:


#we are creating the dummy variables of column 'smoking':0-NO,1-YES
smoking_y=pd.get_dummies(df['smoking'],drop_first=True)
smoking_y


# In[83]:


#dropping the columns that contained the catagorical value
lst=['sex','smoking']
df.drop(lst,axis=1,inplace=True)


# In[84]:


#displaying the structure of the dataset to make sure that the columns have been successfully dropped
df.info()


# In[85]:


#inserting the series in our dataset that contains the dummy numeric value of the column 'sex' and renaming it as 'sex'
df.insert(0, "sex",sex_m,True) 


# In[86]:


#inserting the series in our dataset that contains the dummy numeric value of the column 'smoking' and renaming it as 'smoking'
df.insert(2, "smoking",smoking_y, True) 


# In[87]:


#displaying the structure of the dataset to make sure that the series have been successfully inserted
df.info()


# In[88]:


#displaying the statistical data of our dtaset
df.describe()


# In[89]:


#for a normal distribution,the standard deviation should be 1
#here the columns 'age','chol','max_hrt','bmi','pulse_rate'
#have a very high standard deviation value so we need to transform them


# In[90]:


#plotting the histogram of column 'age' to check its skewness
df['age'].plot.hist(bins=20,figsize=(20,5))


# In[91]:


#as the data is skewed,we are using square root transformation to remove the skewness
age_sqrt = df.age**(1/2)
age_sqrt.describe()


# In[92]:


#now the standard deviation is near 1,so the skewness is removed


# In[93]:


#plotting the histogram of column 'chol' to check its skewness
df['chol'].plot.hist(bins=20,figsize=(20,5))


# In[94]:


#as the data is skewed,we are using square root transformation to remove the skewness
chol_sqt= df.chol**(1/2)
chol_sqt.describe()


# In[95]:


#now the standard deviation is near 1,so the skewness is removed


# In[96]:


#plotting the histogram of column 'bmi' to check its skewness
df['bmi'].plot.hist(bins=20,figsize=(20,5))


# In[97]:


#as the data is skewed,we are using square root transformation to remove the skewness
bmi_s = df.bmi**(1/2)
bmi_s.describe()


# In[98]:


#now the standard deviation is near 1,so the skewness is removed


# In[99]:


#plotting the histogram of column 'max_hrt' to check its skewness
df['max_hrt'].plot.hist(bins=20,figsize=(20,5))


# In[100]:


#as the data is skewed,we are using square root transformation to remove the skewness
max_hrt_st= df.max_hrt**(1/2)
max_hrt_st.describe()


# In[101]:


#now the standard deviation is near 1,so the skewness is removed


# In[102]:


#plotting the histogram of column 'pulse_rate' to check its skewness
df['pulse_rate'].plot.hist(bins=20,figsize=(20,5))


# In[103]:


#as the data is skewed,we are using square root transformation to remove the skewness
pulse_rate_srt= df.pulse_rate**(1/2)
pulse_rate_srt.describe()


# In[104]:


#now the standard deviation is near 1,so the skewness is removed


# In[105]:


#removing the columns with the skewed data
lst=['age','chol','max_hrt','bmi','pulse_rate']


# In[106]:


df.drop(lst,axis=1,inplace=True)


# In[107]:


#displaying the data structure to make sure that the columns have been successfully dropped
df.info()


# In[108]:


#inserting the series with the transformed data into the dataset
df.insert(2, "chol",chol_sqt,True) 


# In[109]:


df.insert(1, "age",age_sqrt,True)


# In[110]:


df.insert(5, "max_hrt",max_hrt_st,True)


# In[111]:


df.insert(6, "bmi",bmi_s,True)


# In[112]:


df.insert(7,"pulse_rate",pulse_rate_srt,True)


# In[113]:


#displaying the structure of the dataset to make sure that our transformed data is successfully added
df.info()


# In[114]:


df['LAD'].value_counts()


# In[115]:


#replacing the NaN values with literal "NA"
df["LAD"].fillna('NA', inplace = True) 


# In[116]:


#displaying the values along with the frequency of column 'LAD'
df['LAD'].value_counts()


# In[117]:


#as the data is catagrical,we are using label encoding to convert it into numeric values
label_encoder = preprocessing.LabelEncoder()  
df['LAD']= label_encoder.fit_transform(df['LAD']) 
df['LAD'].value_counts()


# In[118]:


#replacing the NaN values with literal "NA"
df["CA"].fillna('NA', inplace = True)
#displaying the values along with the frequency of column 'CA'
df['CA'].value_counts()


# In[119]:


#as the data is catagrical,we are using label encoding to convert it into numeric values
df['CA']= label_encoder.fit_transform(df['CA']) 
df['CA'].value_counts()


# In[120]:


#replacing the NaN values with literal "NA"
df["RCA"].fillna('NA', inplace = True)
#displaying the values along with the frequency of column 'CA'
df['RCA'].value_counts()


# In[121]:


#as the data is catagrical,we are using label encoding to convert it into numeric values
df['RCA']= label_encoder.fit_transform(df['RCA']) 
df['RCA'].value_counts()


# In[122]:


#replacing the NaN values with literal "NA"
df["OM"].fillna('NA', inplace = True)
#displaying the values along with the frequency of column 'OM'
df['OM'].value_counts()


# In[123]:


#as the data is catagrical,we are using label encoding to convert it into numeric values
df['OM']= label_encoder.fit_transform(df['OM']) 
df['OM'].value_counts()


# In[124]:


#replacing the NaN values with literal "NA"
df["LPD"].fillna('NA', inplace = True)
#displaying the values along with the frequency of column 'LPD'
df['LPD'].value_counts()


# In[125]:


#as the data is catagrical,we are using label encoding to convert it into numeric values
df['LPD']= label_encoder.fit_transform(df['LPD']) 
df['LPD'].value_counts()


# In[126]:


#replacing the NaN values with literal "NA"
df["LCA"].fillna('NA', inplace = True)
#displaying the values along with the frequency of column 'LCA'
df['LCA'].value_counts()


# In[127]:


#as the data is catagrical,we are using label encoding to convert it into numeric values
df['LCA']= label_encoder.fit_transform(df['LCA']) 
df['LCA'].value_counts()


# In[128]:


#as we have converted the data into numeric form,we are checking the statistical data of the columns
df.describe()


# In[129]:


#the standard deviation of the column 'OM' is high,we need to transform it
#plotting the histogram of column 'pulse_rate' to check its skewness
df['OM'].plot.hist(bins=20,figsize=(20,5))


# In[130]:


#as the data is skewed,we are using square root transformation to remove the skewness
OM_st= df.OM**(1/2)
OM_st.describe()


# In[131]:


#now the standard deviation is near 1,so the skewness is removed


# In[132]:


#dropping the column consisting of skewed data
df.drop('OM',axis=1,inplace=True)


# In[133]:


#inserting the series containing the transformed data into the dataset
df.insert(15, "OM",OM_st,True) 


# In[ ]:





# In[134]:


#displayong the dataset structure to make sure the necessary changes are done
df.info()


# In[135]:


#dropping the unnamed column consisting of all NaN values
df.drop('Unnamed: 25',axis=1,inplace=True)


# In[136]:


#dropping the columns consisting of NaN values
df.dropna(inplace=True)


# In[ ]:





# In[137]:


df.isnull().sum()


# In[138]:


#checking all the unique values of column 'sex' to make sure its numeric
df['sex'].unique()


# In[139]:


#checking all the unique values of column 'sex' to make sure its numeric
df['age'].unique()


# In[140]:


#checking all the unique values of column 'smoking' to make sure its numeric
df['smoking'].unique()


# In[141]:


#checking all the unique values of column 'chol' to make sure its numeric
df['chol'].unique()


# In[142]:


#checking all the unique values of column 'chest pain' to make sure its numeric
df['chest pain'].unique()


# In[143]:


#checking all the unique values of column 'max_hrt' to make sure its numeric
df['max_hrt'].unique()


# In[144]:


#checking all the unique values of column 'bmi' to make sure its numeric
df['bmi'].unique()


# In[145]:


#checking all the unique values of column 'pulse_rate' to make sure its numeric
df['pulse_rate'].unique()


# In[146]:


#checking all the unique values of column 'rbp' to make sure its numeric
df['rbp'].unique()


# In[147]:


#the values of column 'rbp' are catagorical


# In[148]:


#checking all the unique values of column 'fbs' to make sure its numeric
df['fbs'].unique()


# In[149]:


#checking all the unique values of column 'recg' to make sure its numeric
df['recg'].unique()


# In[150]:


#checking all the unique values of column 'oldpeak' to make sure its numeric
df['oldpeak'].unique()


# In[151]:


#the values of the column 'oldpeak' is catagorical


# In[152]:


#checking all the unique values of column 'slope' to make sure its numeric
df['slope'].unique()


# In[153]:


#checking all the unique values of column 'exang' to make sure its numeric
df['exang'].unique()


# In[154]:


#checking all the unique values of column 'LAD' to make sure its numeric
df['LAD'].unique()


# In[155]:


#checking all the unique values of column 'OM' to make sure its numeric
df['OM'].unique()


# In[156]:


#checking all the unique values of column 'CA' to make sure its numeric
df['CA'].unique()


# In[157]:


#checking all the unique values of column 'RCA' to make sure its numeric
df['RCA'].unique()


# In[158]:


#checking all the unique values of column 'LPD' to make sure its numeric
df['LPD'].unique()


# In[159]:


#checking all the unique values of column 'LCA' to make sure its numeric
df['LCA'].unique()


# In[160]:


#checking all the unique values of column 'stent' to make sure its numeric
df['stent'].unique()


# In[161]:


#since the data values of column 'rbp' was catagorical,we converted it into numeric data using line encoding
df['rbp']= label_encoder.fit_transform(df['rbp']) 
df['rbp'].value_counts()


# In[162]:


#since the data values of column 'oldpeak' was catagorical,we converted it into numeric data using line encoding
df['oldpeak']= label_encoder.fit_transform(df['oldpeak']) 
df['oldpeak'].value_counts()


# In[163]:


df.info()


# In[164]:


#now all the values are in numeric form,we can calculate its f-value and probability to select the most relevant columns


# In[165]:


#X:input data or the independent variables
#y:output data or the dependent variable


# In[166]:


X = df.drop('stent',axis=1)
y=df['stent']


# In[167]:


#calculating the f-value and probability of the columns
di={}
fvalue,prob=feature_selection.f_classif(X,y)
for col,f,p in zip(X.columns,fvalue,prob):
    di[col]=[f,p]
f_p=pd.DataFrame(di)
f_p.rename(index={0:"F-value",1:"Probability"})


# In[168]:


#feature_selection module calculates f-values and probabilities of every column
#Based on the f-values and probabilities, we got an idea of columns to be dropped
#Higher the f-value and lower the probability, higher is the influence of that column on the dependent variable
lst=['fbs','rbp','LCA','LPD','pulse_rate','recg']


# In[169]:


df.drop(lst,axis=1,inplace=True)


# In[170]:


#displaying the structure of the dataset to make sure the changes are done
df.info()


# In[171]:


X = df.drop('stent',axis=1)
y=df['stent']


# In[172]:


#splitting the dataset into train and test
#the train set will be used to train the model
#the test set will be used to test the model
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


# In[173]:


#decision tree classifier
from sklearn.tree import DecisionTreeClassifier 
# training a DescisionTreeClassifier 
from sklearn import metrics
dtree = DecisionTreeClassifier().fit(X_train, y_train) 
dtree_predictions = dtree.predict(X_test) 
# accuracy on X_test 
accuracy = dtree.score(X_test, y_test) 
print("accuracy:")
print(accuracy)
#creating the confusion matrix
 
print(metrics.confusion_matrix(y_test,dtree_predictions))
#the classification report is printed
print(metrics.classification_report(y_test, dtree_predictions))


# In[174]:


#knn classifier
from sklearn.neighbors import KNeighborsClassifier 
#training the knn classifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 

# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print("accuracy:")
print(accuracy)

# creating a confusion matrix 
knn_predictions = knn.predict(X_test) 
#creating the confusion matrix
print(metrics.confusion_matrix(y_test,knn_predictions))
#the classification report is printed
print(metrics.classification_report(y_test, knn_predictions))


# In[175]:


#naive bayes

# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 
print("accuracy:")
print(accuracy)

#creating the confusion matrix
print(metrics.confusion_matrix(y_test,gnb_predictions))
#the classification report is printed
print(metrics.classification_report(y_test, gnb_predictions))


# In[176]:


from sklearn.ensemble import RandomForestClassifier
#training the random forest classifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
forest.fit(X_train, y_train)
rnd_predictions = forest.predict(X_test) 
# accuracy on X_test 
accuracy = forest.score(X_test, y_test) 
print("accuracy:")
print(accuracy)
#creating the confusion matrix
print(metrics.confusion_matrix(y_test,rnd_predictions))
#the classification report is printed
print(metrics.classification_report(y_test, rnd_predictions))


# In[177]:


#as the decision tree classifier model has the maximum accuracy,it shall be used to implement the model


# In[178]:


#help('modules')


# In[179]:


#import joblib
#dtree_model=open("dtree_model.pkl","wb")
#joblib.dump(dtree,dtree_model)


# In[180]:


#dtree_model.close()


# In[181]:


df.to_csv('stent.csv')


# In[182]:


#pip install ipython


# In[183]:


#pip install nbconvert


# In[184]:


#ipython nbconvert-to script enigma.ipynb


# In[185]:


#ipython nbconvert — to script enigma.ipynb


# In[187]:


#get_ipython().system('jupyter nbconvert --to script MISSION DELLOITE.ipynb')


# In[188]:


#pip install ipynb-py-convert


# In[189]:


#get_ipython().system('jupyter nbconvert --to script MISSION DELLOITE.ipynb')


# In[190]:


#get_ipython().system('jupyter nbconvert --to script enigma.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




