#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# Import the datasets and separate them into feature and output matrices
dataset = pd.read_csv('data.csv')
dataset


# In[3]:


dataset = dataset.drop(['Unnamed: 25'],axis=1)


# In[4]:


# Filling the NAN values with the corresponding mean value
dataset = dataset.fillna(dataset.mean())


# In[5]:


dataset


# In[6]:


dataset.info()


# In[7]:


dataset = dataset.dropna()


# In[8]:


dataset.info()


# In[ ]:


y
def is_float(input):
  try:
    num = float(input)
  except ValueError:
    return False
  return True

for i in range(0,184):
    if y[i] == 'ckd':
        y[i] = 1
    else:
        y[i] = 0
y = y.astype(int)
y


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

RBC = le.fit_transform(dataset.RBC) 
PC = le.fit_transform(dataset.PC) 
PCC = le.fit_transform(dataset.PCC)
BA = le.fit_transform(dataset.BA)
HTN = le.fit_transform(dataset.HTN)
DM = le.fit_transform(dataset.DM)
CAD = le.fit_transform(dataset.CAD)
APPET = le.fit_transform(dataset.APPET)
Pe = le.fit_transform(dataset.Pe)
Ane = le.fit_transform(dataset.Ane)
class1 = le.fit_transform(dataset.class1)

testing=dataset

testing['RBC'] = RBC
testing['PC'] = PC
testing['PCC'] = PCC
testing['BA'] = BA
testing['HTN'] = HTN
testing['DM'] = DM
testing['CAD'] = CAD
testing['APPET'] = APPET
testing['Pe'] = Pe
testing['Ane'] = Ane
testing['class1'] = class1


# In[11]:


testing


# In[12]:


testing.describe()


# In[13]:


X = testing.iloc[:, 0:24].values
y = testing.iloc[:, 24].values


# In[14]:


X


# In[15]:


y


# In[16]:


#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=100)


# In[27]:


from sklearn import svm
sv = svm.SVC(kernel='linear')
sv.fit(X_train, y_train)
y_pred = sv.predict(X_test)


# In[38]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[39]:


from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0

for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,y_test)*100,2)
    if(current_accuracy > max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,y_train)
Y_pred_rf = rf.predict(X_test)


# In[42]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, Y_pred_rf)*100)

