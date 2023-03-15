#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("E:\\data science\\train (2).csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df['age'].unique()


# In[6]:


df['job'].unique(
)


# In[7]:


df['marital'].unique()


# In[8]:


df['education_qual'].unique()


# In[9]:


df['call_type'].unique()


# In[10]:


df['day'].unique()


# In[11]:


df['mon'].unique()


# In[12]:


df['dur'].unique()


# In[13]:


df['num_calls'].unique()


# In[14]:


df['prev_outcome'].unique()


# In[15]:


df=df.drop_duplicates()


# In[16]:


df['education_qual'].value_counts()


# In[17]:


df['education_qual']=df['education_qual'].replace('unknown',df['education_qual'].mode()[0])


# In[18]:


df['education_qual'].value_counts()


# In[19]:


df['job'].value_counts()


# In[20]:


df['job']=df['job'].replace('unknown',df['job'].mode()[0])


# In[21]:


df['job'].value_counts()


# In[22]:


df.marital.value_counts()


# In[23]:


df.call_type.value_counts()


# In[24]:


df.mon.value_counts()


# In[25]:


df.prev_outcome.value_counts()


# In[26]:


df.y.value_counts()


# In[27]:


df.isnull().sum()


# In[28]:


df.shape


# In[29]:


df.describe()


# In[30]:


IQR=df['dur'].quantile(0.75)-df['dur'].quantile(0.25)
UL=df.dur.quantile(0.75)+(1.5*IQR)
LL=df.dur.quantile(0.25)-(1.5*IQR)
UL,LL


# In[31]:


df.dur=df.dur.clip(LL,UL)


# In[32]:


IQR=df['num_calls'].quantile(0.75)-df['num_calls'].quantile(0.25)
UL=df.num_calls.quantile(0.75)+(1.5*IQR)
LL=df.num_calls.quantile(0.25)-(1.5*IQR)
UL,LL


# In[33]:


df.num_calls=df.num_calls.clip(LL,UL)


# In[34]:


df.describe()


# In[35]:


df['target']=df['y'].map({'yes':1,'no':0})


# In[36]:


df.head()


# In[37]:


df.groupby('prev_outcome')['target'].mean()


# In[38]:


df.info()


# In[39]:


df.dtypes


# In[40]:


import warnings
warnings.filterwarnings('ignore')


# EDA-EXPLORATORY DATA ANALYSIS

# In[41]:


df_i=pd.DataFrame(df.job.value_counts()).sort_values('job',ascending=False).reset_index()
df_i.rename(columns={'index':'jobb','job':'count'},inplace=True)
bar=sns.barplot(x=df_i['jobb'],y=df_i['count'],data=df_i)
bar.tick_params(axis='x',rotation=90)


# In[42]:


i=sns.countplot(df,x='job',hue='y')
i.tick_params(axis='x',rotation=90)


# In[43]:


(df.groupby('job')['target'].mean()*100).sort_values().plot(kind='barh',color='blue')


# In[44]:


(df.groupby('marital')['target'].mean()*100).sort_values().plot(kind='barh',color='pink')


# In[45]:


(df.groupby('education_qual')['target'].mean()*100).sort_values().plot(kind='barh',color='orange')


# In[46]:


(df.groupby('call_type')['target'].mean()*100).sort_values().plot(kind='barh',color='blue')


# In[47]:


(df.groupby('mon')['target'].mean()*100).sort_values().plot(kind='barh',color='yellow')


# In[48]:


(df.groupby('prev_outcome')['target'].mean()*100).sort_values().plot(kind='barh',color='gray')


# ENCODING
# 

# In[49]:


col=df['job'].unique()
P=[]
for i in col:
 p=len(df[df['job']==i][df['y']=='yes'])/len(df[df['job']==i])
 P.append(p)
dff=pd.DataFrame({'job':col,'%':P})
dff=dff.sort_values('%',ascending=True)
dff=dff.reset_index()
del dff['index']


# In[50]:


dff


# In[51]:


df['job']=df['job'].map({dff['job'][0]:0,dff['job'][1]:1,dff['job'][2]:2,dff['job'][3]:3,dff['job'][4]:4,dff['job'][5]:5,dff['job'][6]:6,dff['job'][7]:7,dff['job'][8]:8,dff['job'][9]:9,dff['job'][10]:10})


# In[52]:


col =df['marital'].unique()
P=[]
for i in col:
 p=len(df[df['marital']==i][df['y']=='yes'])/len(df[df['marital']==i])
 P.append(p)
dff=pd.DataFrame({'marital':col,'%':P})
dff=dff.sort_values('%',ascending=True)
dff=dff.reset_index()
del dff['index']


# In[53]:


dff


# In[54]:


df['marital']=df['marital'].map({dff['marital'][0]:0,dff['marital'][1]:1,dff['marital'][2]:2})


# In[55]:


col =df['education_qual'].unique()
P=[]
for i in col:
 p=len(df[df['education_qual']==i][df['y']=='yes'])/len(df[df['education_qual']==i])
 P.append(p)
dff=pd.DataFrame({'education_qual':col,'%':P})
dff=dff.sort_values('%',ascending=True)
dff=dff.reset_index()
del dff['index']
dff


# In[56]:


df['education_qual']=df['education_qual'].map({dff['education_qual'][0]:0,dff['education_qual'][1]:1,dff['education_qual'][2]:2})


# In[57]:


col =df['call_type'].unique()
P=[]
for i in col:
 p=len(df[df['call_type']==i][df['y']=='yes'])/len(df[df['call_type']==i])
 P.append(p)
dff=pd.DataFrame({'call_type':col,'%':P})
dff=dff.sort_values('%',ascending=True)
dff=dff.reset_index()
del dff['index']
dff


# In[58]:


df['call_type']=df['call_type'].map({dff['call_type'][0]:0,dff['call_type'][1]:1,dff['call_type'][2]:2})


# In[59]:


col =df['mon'].unique()
P=[]
for i in col:
 p=len(df[df['mon']==i][df['y']=='yes'])/len(df[df['mon']==i])
 P.append(p)
dff=pd.DataFrame({'mon':col,'%':P})
dff=dff.sort_values('%',ascending=True)
dff=dff.reset_index()
del dff['index']
dff


# In[60]:


df['mon']=df['mon'].map({dff['mon'][0]:0,dff['mon'][1]:1,dff['mon'][2]:2,dff['mon'][3]:3,dff['mon'][4]:4,dff['mon'][5]:5,dff['mon'][6]:6,dff['mon'][7]:7,dff['mon'][8]:8,dff['mon'][9]:9,dff['mon'][10]:10,dff['mon'][11]:11})


# In[61]:


col =df['prev_outcome'].unique()
P=[]
for i in col:
 p=len(df[df['prev_outcome']==i][df['y']=='yes'])/len(df[df['prev_outcome']==i])
 P.append(p)
dff=pd.DataFrame({'prev_outcome':col,'%':P})
dff=dff.sort_values('%',ascending=True)
dff=dff.reset_index()
del dff['index']
dff


# In[62]:


df['prev_outcome']=df['prev_outcome'].map({dff['prev_outcome'][0]:0,dff['prev_outcome'][1]:1,dff['prev_outcome'][2]:2,dff['prev_outcome'][3]:3})


# In[63]:


df.head()


# SPLITTING
# 

# In[64]:


col=[*df.columns]
col[:-2]


# In[65]:


x=df.loc[:,col[:-2]].values
y=df.loc[:,col[-1]].values


# In[66]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# BALANCING DATA
# 

# In[67]:


df.shape


# In[68]:


len(x_train),len(y_train)


# In[69]:


from imblearn.combine import SMOTEENN
smt=SMOTEENN(sampling_strategy='all')
x_smt,y_smt=smt.fit_resample(x_train,y_train)


# In[70]:


len(x_smt),len(y_smt)


# In[71]:


df_bal=pd.DataFrame(x_smt,columns=df.columns[:-2])
df_bal['y']=y_smt


# In[72]:


len(df_bal[df_bal['y']==1])/len(df_bal)


# SCALING OF DATASET
# 

# In[73]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_smt)
x_test_scaled=scaler.transform(x_test)


# In[74]:


x_train_scaled


# In[75]:


x_test_scaled


# MODEL: Logistis Regression
# 

# In[76]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

lr=LogisticRegression()
lr.fit(x_train_scaled,y_smt)
lr.score(x_test_scaled,y_test)
log=roc_auc_score(y_test,lr.predict_proba(x_test_scaled)[:, 1])


# MODEL: KNN
# 

# In[77]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier()
KNN.fit(x_train_scaled,y_smt)
KNN.score(x_test_scaled,y_test)


# In[78]:


'''for i in [1,2,3,4,5,6,7,8,9,10,20,30,40,50]:
 kNN=KNeighborsClassifier(n_neighbors=i)
 KNN.fit(x_train_scaled,y_smt)
 print('K-Values:',i,"Accuracy score:",KNN.score(x_train_scaled,y_smt),'Cross-val score:',np.mean(cross_val_score(KNN,x_train_scaled,y_smt,cv=10)))''''''


# In[79]:


KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train_scaled,y_smt)
KNN.score(x_test_scaled,y_test)
k=roc_auc_score(y_test,KNN.predict_proba(x_test_scaled)[:, 1])


# MODEL: Decision Tree
# 

# In[80]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train_scaled,y_smt)
dt.score(x_test_scaled,y_test)


# In[81]:


'''for d in [1,2,3,4,5,6,7,8,9,10]:
 dtt=DecisionTreeClassifier(max_depth=d)
 dtt.fit(x_train_scaled,y_smt)
 tt=DecisionTreeClassifier(max_depth=d)
 from sklearn.metrics import accuracy_score
 print('depth:',d,'Accuracy score:',accuracy_score(y_smt,dtt.predict(x_train_scaled)),'cv:',np.mean(cross_val_score(tt, x_train_scaled, y_smt, cv=10)))'''


# In[82]:


dtt=DecisionTreeClassifier(max_depth=10)
dtt.fit(x_train_scaled,y_smt)
dt.score(x_test_scaled,y_test)
d=roc_auc_score(y_test,dtt.predict_proba(x_test_scaled)[:, 1])


# In[83]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,dtt.predict(x_test_scaled))


# MODEL: Random Forest

# In[84]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,max_depth=4,max_features='sqrt')
rf.fit(x_train_scaled,y_smt)


# In[85]:


from sklearn.metrics import roc_auc_score
r=roc_auc_score(y_test,dtt.predict_proba(x_test_scaled)[:, 1])


# In[86]:


confusion_matrix(y_test,rf.predict(x_test_scaled))


# MODEL: XGBoost

# In[87]:


import xgboost as xgb


# In[88]:


'''for lr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.10,0.11,0.12]:
    xg=xgb.XGBClassifier(learning_rate=lr,n_estimators=100,verbosity=0)
    xg.fit(x_train_scaled,y_smt)
    xg.score(x_test_scaled,y_test)
    print('LR:',lr,'train score:',xg.score(x_train_scaled,y_smt),'c v:',np.mean(cross_val_score(xg,x_train_scaled,y_smt,cv=10)))'''


# In[89]:


xg=xgb.XGBClassifier(learning_rate=lr,n_estimators=100,verbosity=0)
xg.fit(x_train_scaled,y_smt)
g=roc_auc_score(y_test,xg.predict_proba(x_test_scaled)[:, 1])


# In[90]:


xg.score(x_train_scaled,y_smt)


# In[91]:


confusion_matrix(y_test,xg.predict(x_test_scaled))


# VOTING CLASSIFIER
# 

# In[92]:


from sklearn.ensemble import VotingClassifier
from sklearn import tree
m1=LogisticRegression(random_state=12)
m2=tree.DecisionTreeClassifier(random_state=12)
m3=KNeighborsClassifier(5)
m4=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
m5=RandomForestClassifier(n_estimators=100,max_depth=5,max_features='sqrt')
m=VotingClassifier(estimators=[('lr',m1),('dt',m2),('knn',m3),('xgb',m4),('rf',m5)],voting='soft')


# In[93]:


m.fit(x_train_scaled,y_smt)
y_pred=m.predict(x_test_scaled)
v=roc_auc_score(y_test,m.predict_proba(x_test_scaled)[:,1])


# In[94]:


pd.DataFrame({'Model':['Logistic Regression','knn','Decision Tree','Random Forest','xgboost','voting Classifier'],'AUROC':[log,k,d,r,g,v]})


# In[95]:


imp_ft=pd.DataFrame({'ft':col[:-2], 'imp':xg.feature_importances_})
imp_ft.sort_values('imp',ascending=False,inplace=True)


# In[96]:


imp_ft.iloc[0:5,0].values


# In[97]:


x_imp=df.loc[:,imp_ft.iloc[0:5,0]].values
y=df.loc[:,col[-1]].values


# In[98]:


from sklearn.model_selection import train_test_split
x_train_imp,x_test_imp,y_train,y_test=train_test_split(x_imp,y,test_size=0.25)


# In[99]:


from imblearn.combine import SMOTEENN
smt=SMOTEENN(sampling_strategy='all')
x_smt_imp,y_smt=smt.fit_resample(x_train_imp,y_train)


# In[100]:


df_bal_imp=pd.DataFrame(x_smt_imp,columns=imp_ft.iloc[0:5,0])
df_bal_imp['y']=y_smt
len(df_bal_imp[df_bal_imp['y']==1])/len(df_bal_imp)


# In[101]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_imp_scaled=scaler.fit_transform(x_smt_imp)
x_test_imp_scaled=scaler.transform(x_test_imp)


# In[102]:


'''for lr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.10,0.11,0.12]:
    xg=xgb.XGBClassifier(learning_rate=lr,n_estimators=100,verbosity=0)
    xg.fit(x_train_imp_scaled,y_smt)
    xg.score(x_test_imp_scaled,y_test)
    print('LR:',lr,'train score:',xg.score(x_train_imp_scaled,y_smt),'c v:',np.mean(cross_val_score(xg,x_train_imp_scaled,y_smt,cv=10)))'''


# In[103]:


xg=xgb.XGBClassifier(learning_rate=lr,n_estimators=100,verbosity=0)
xg.fit(x_train_imp_scaled,y_smt)
g=roc_auc_score(y_test,xg.predict_proba(x_test_imp_scaled)[:, 1])


# In[104]:


from itertools import combinations


# In[117]:


comb_1=list(combinations(col[:-2],1))
comb_1
[comb_1[0]]


# In[106]:


len(col[:-2])+1


# In[116]:


comb_1=list(combinations(col[:-2],1))
auc=[]
for i in comb_1:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    #xgg.score(x_test_scaled,y_test)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc.append(g)
    


# In[118]:


comb_2=list(combinations(col[:-2],2))
auc2=[]
for i in comb_2:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc2 .append(g)
    


# In[ ]:


comb_3=list(combinations(col[:-2],3))
auc3=[]
for i in comb_3:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc3.append(g)
    


# In[ ]:


comb_4=list(combinations(col[:-2],4))
auc4=[]
for i in comb_4:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc4.append(g)
    


# In[ ]:


comb_5=list(combinations(col[:-2],5))
auc5=[]
for i in comb_5:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc5.append(g)
    


# In[ ]:


comb_6=list(combinations(col[:-2],6))
auc6=[]
for i in comb_6:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc6.append(g)
    


# In[ ]:


comb_7=list(combinations(col[:-2]7))
auc7=[]
for i in comb_7:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc7.append(g)
    


# In[ ]:


comb_8=list(combinations(col[:-2],8))
auc8=[]
for i in comb_8:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc8.append(g)
    


# In[ ]:


comb_9=list(combinations(col[:-2],9))
auc9=[]
for i in comb_9:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc9.append(g)
    


# In[ ]:


comb_10=list(combinations(col[:-2],10))
auc10=[]
for i in comb_10:
    x=df.loc[:,i].values
    y=df.loc[:,col[-1]].values
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    smt=SMOTEENN(sampling_strategy='all')
    x_smt,y_smt=smt.fit_resample(x_train,y_train)
    scaler=StandardScaler()
    x_train_scaled=scaler.fit_transform(x_smt)
    x_test_scaled=scaler.transform(x_test)
    import xgboost as xgb
    xgg=xgb.XGBClassifier(learning_rate=0.75,n_estimators=100,verbosity=0)
    xgg.fit(x_train_scaled,y_smt)
    g=roc_auc_score(y_test,xgg.predict_proba(x_test_scaled)[:, 1])
    auc10.append(g)
    


# In[ ]:





# In[ ]:


m=max(auc7)
i=auc7.index(m)
A=com_7[i]
A


# In[ ]:


m


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




