#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


feature_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack_types", 'score'])

R2L=['warezmaster','warezclient','spy','phf','multihop','imap','guess_passwd','ftp_write']
U2R=['rootkit','perl','loadmodule','buffer_overflow']
DoS=['smurf','teardrop','back','land','neptune','pod']
Probe=['ipsweep','nmap','portsweep','satan']


# In[43]:


labelNames=['Normal', 'R2L', 'U2R', 'DoS', 'Probe']
train_df = pd.read_csv(r'/content/drive/MyDrive/NSL_KDD-master/KDDTrain+.csv', names=feature_names)
train_df


# In[44]:


test_df = pd.read_csv(r'/content/drive/MyDrive/NSL_KDD-master/KDDTest+.csv', names=feature_names)
test_df


# In[45]:


def labelAssignment(col):
    attackType=[]
    for sc in col:
        if sc == "normal":
            attackType.append(0) # 0:'Normal'
        elif sc in R2L:
            attackType.append(1) # 1:'R2L'
        elif sc in U2R:
            attackType.append(2) # 2:'U2R'
        elif sc in DoS:
            attackType.append(3) # 3:'DoS'
        else:
            attackType.append(4) # 4:'Probe'
    return attackType


# In[46]:


train_df["label"]= labelAssignment(train_df["attack_types"])

train_df


# In[47]:


test_df["label"]= labelAssignment(test_df["attack_types"])

test_df


# ## Conversion of categorical data

# In[48]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# Service Variable
train_df.service = encoder.fit_transform(train_df.service)
test_df.service = encoder.fit_transform(test_df.service)

# Flag variable
train_df.flag = encoder.fit_transform(train_df.flag)
test_df.flag = encoder.fit_transform(test_df.flag)

# protocol type variable
train_df.protocol_type = encoder.fit_transform(train_df.protocol_type)
test_df.protocol_type = encoder.fit_transform(test_df.protocol_type)


# In[49]:


# train_df.duration = (train_df.duration - train_df.duration.min()) / (train_df.duration.max() - train_df.duration.min())    
# train_df.src_bytes = (train_df.src_bytes - train_df.src_bytes.min()) / (train_df.src_bytes.max() - train_df.src_bytes.min())    
# train_df.dst_bytes = (train_df.dst_bytes - train_df.dst_bytes.min()) / (train_df.dst_bytes.max() - train_df.dst_bytes.min())    
# a = [10, 10, 10, 10, 10, 10, 20, 30, 40, 50, 60, 70]

train_df.duration = encoder.fit_transform(pd.cut(train_df.duration, bins=6, precision=0))
# f = pd.cut(train_df.src_bytes, bins=6, precision=0)
# f = pd.cut(train_df.dst_bytes, bins=6, precision=0)
# train_df.duration = encoder.fit_transform(train_df.duration)
train_df.duration.value_counts(), len(np.unique(train_df.src_bytes)), len(np.unique(train_df.dst_bytes))


# In[50]:


del train_df["attack_types"]
del train_df["score"]
del test_df["attack_types"]
del test_df["score"]


# In[51]:


print('Before remove duplicate shape',train_df.shape)
train_df.drop_duplicates(subset=None, keep='first', inplace=True)
print('After remove duplicate shape',train_df.shape)


# ## Checking the distribution of classes

# In[52]:


train_df["label"].value_counts()


# ## Splitting datasets into feautures and labels

# In[53]:


train_X = train_df.iloc[:,:-1]
train_y = train_df.label

test_X = test_df.iloc[:,:-1]
test_y = test_df.label


# In[54]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size = 0.2)


# # To prove our hypothesis

# In[55]:


X_test_train, X_test_valid, y_test_train, y_test_valid = train_test_split(test_X, test_y, test_size = 0.2)


# In[56]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, plot_confusion_matrix

rf_clf_test = RandomForestClassifier(random_state=1).fit(X_test_train, y_test_train)
print(rf_clf_test.score(X_test_valid, y_test_valid), rf_clf_test.score(X_train, y_train))


# In[57]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(rf_clf_test, X_test_valid, y_test_valid, ax=ax, normalize='true', cmap="summer")


# In[58]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(rf_clf_test, X_train, y_train, ax=ax, normalize='true', cmap="summer")


# # Creating and comparing models

# ## Decision Tree

# In[ ]:


from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)

print(dt_clf.score(X_valid, y_valid), dt_clf.score(test_X, test_y))


# ## AdaBoost
# 

# In[ ]:


from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

ab_clf = AdaBoostClassifier(n_estimators=100, random_state=1).fit(X_train, y_train)

print(ab_clf.score(X_valid, y_valid), ab_clf.score(test_X, test_y))


# ## XGBoost

# In[ ]:


from xgboost import XGBClassifier

xgb_clf = XGBClassifier().fit(X_train, y_train)
print(xgb_clf.score(X_valid, y_valid), xgb_clf.score(test_X, test_y))


# ## Random Forest

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, plot_confusion_matrix
                            
rf_clf = RandomForestClassifier(random_state=1).fit(X_train, y_train)


# ### Validation set

# In[ ]:


rf_clf.score(X_valid, y_valid)


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(rf_clf, X_valid, y_valid, ax=ax, cmap="summer")


# ### Test set

# In[ ]:


rf_clf.score(test_X, test_y)


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(rf_clf, test_X, test_y, ax=ax, cmap="summer")


# ## KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier().fit(X_train, y_train)


# ### Validation set

# In[ ]:


knn_clf.score(X_valid, y_valid)


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(knn_clf, X_valid, y_valid, ax=ax, cmap="summer")


# ### Test set

# In[ ]:


knn_clf.score(test_X, test_y)


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(knn_clf, test_X, test_y, ax=ax, cmap="summer")


# ## Extra Trees

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
                            
et_clf = ExtraTreesClassifier(random_state=1).fit(X_train, y_train)


# ### Validation set

# In[ ]:


et_clf.score(X_valid, y_valid)


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(et_clf, X_valid, y_valid, ax=ax, cmap="summer")


# ### Test set

# In[ ]:


et_clf.score(test_X, test_y)


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(et_clf, test_X, test_y, ax=ax, cmap="summer")


# # Ensemble Learning

# ## Solving oversampling problem

# In[60]:


from imblearn.over_sampling import SMOTE

oversample = SMOTE()

train_X, train_y = oversample.fit_resample(train_X, train_y)


# ## Splitting the data into three parts

# In[61]:


model1_train, model2_train, model1_labels, model2_labels = train_test_split(np.array(train_X), np.array(train_y), test_size = 0.33, shuffle=True, random_state=1)
model1_train, model3_train, model1_labels, model3_labels = train_test_split(model1_train, model1_labels, test_size = 0.5, shuffle=True, random_state=1)

len(model1_labels), len(model2_labels), len(model3_labels)


# In[62]:


from sklearn.metrics import accuracy_score

def accuracy_calculation(estimator, test_data, actual_labels):
    predictions = estimator.predict(test_data)

    return predictions, accuracy_score(actual_labels, predictions)


# # Trained Models

# ## model 4

# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[3, 5, 7], 'n_estimators':[30, 40, 50], 'seed':[1]}
xgb_clf = XGBClassifier()

model2 = GridSearchCV(xgb_clf, parameters, verbose=1, n_jobs=-1).fit(model2_train, model2_labels)

test_predictions_2, test_acc = accuracy_calculation(model2, test_X.values, test_y.values)


# In[ ]:


print(model2.best_estimator_, test_acc)


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model2, test_X.values, test_y.values, ax=ax, cmap="summer")


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model2, test_X.values, test_y.values, ax=ax, cmap="summer", normalize='true')


# ## model 5

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[3, 5, 7, 9], 'min_samples_split':[5, 10, 15], 'min_samples_leaf':[5, 10, 15], 'random_state':[1]}
dt_clf = DecisionTreeClassifier().fit(X_train, y_train)

model2 = GridSearchCV(dt_clf, parameters, verbose=1, n_jobs=-1).fit(model2_train, model2_labels)

test_predictions_2, test_acc = accuracy_calculation(model2, test_X.values, test_y.values)
print(model2.best_estimator_, test_acc)


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model2, test_X.values, test_y.values, ax=ax, cmap="summer")


# In[ ]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model2, test_X.values, test_y.values, ax=ax, cmap="summer", normalize='true')


# ## Model 1

# In[63]:


from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[3, 5, 7, 9], 'min_samples_split':[5, 10, 15], 'n_estimators':[40, 50, 60, 70], 'min_samples_leaf':[5, 10, 15], 'random_state':[1]}

rf_clf = RandomForestClassifier() # random_state=1, min_samples_split=10, n_estimators=66, min_samples_leaf=3
model1 = GridSearchCV(rf_clf, parameters, verbose=1, n_jobs=-1).fit(model1_train, model1_labels)

data2_predictions, data2_acc = accuracy_calculation(model1, model2_train, model2_labels)
data3_predictions, data3_acc = accuracy_calculation(model1, model3_train, model3_labels)
test_predictions_1, test_acc = accuracy_calculation(model1, test_X, test_y)

print('Acc with block 2 data ', data2_acc)
print('Acc with block 3 data ', data3_acc)


# In[64]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model1, test_X, test_y, ax=ax, cmap="summer")


# In[65]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model1, test_X, test_y, ax=ax, cmap="summer", normalize="true")


# ## Model 2

# In[66]:


from sklearn.neighbors import KNeighborsClassifier

model2 = KNeighborsClassifier().fit(model2_train, model2_labels)

data1_predictions, data1_acc = accuracy_calculation(model2, model1_train, model1_labels)
data3_predictions, data3_acc = accuracy_calculation(model2, model3_train, model3_labels)
test_predictions_2, test_acc = accuracy_calculation(model2, test_X, test_y)

print('Acc with block 1 data ', data1_acc)
print('Acc with block 3 data ', data3_acc)


# In[67]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model2, test_X, test_y, ax=ax, cmap="summer")


# In[68]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model2, test_X, test_y, ax=ax, cmap="summer", normalize="true")


# ## Model 3

# In[69]:


from sklearn.ensemble import ExtraTreesClassifier

parameters = {'max_depth':[3, 5, 7, 9], 'min_samples_split':[5, 10, 15], 'n_estimators':[40, 50, 60, 70], 'min_samples_leaf':[5, 10, 15], 'random_state':[1]}


et_clf = ExtraTreesClassifier() # random_state=1, min_samples_split=10, n_estimators=68, min_samples_leaf=4
model3 = GridSearchCV(et_clf, parameters, verbose=1, n_jobs=-1).fit(model3_train, model3_labels)

data1_predictions, data1_acc = accuracy_calculation(model3, model1_train, model1_labels)
data2_predictions, data2_acc = accuracy_calculation(model3, model2_train, model2_labels)
test_predictions_3, test_acc = accuracy_calculation(model3, test_X, test_y)

print('Acc with block 1 data ', data1_acc)
print('Acc with block 2 data ', data2_acc)


# In[70]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model3, test_X, test_y, ax=ax, cmap="summer")


# In[71]:


fig, ax = plt.subplots(figsize=(5, 5))
plot_confusion_matrix(model3, test_X, test_y, ax=ax, cmap="summer", normalize="true")


# ## Getting final predictions

# In[72]:


import random

def most_frequent(List):
    return max(set(List), key = List.count)

def get_predictions(pred1, pred2, pred3):
    final_preds = []
    for i in range(0, len(pred1)):
        classes = []
        classes.append(pred1[i])
        classes.append(pred2[i])
        classes.append(pred3[i])

        if pred1[i] != pred2[i] and pred1[i] != pred3[i] and pred2[i] != pred3[i]:
            final_preds.append(classes[2]) # chooses from best model
        else:
            final_preds.append(most_frequent(classes))

    return final_preds


# In[73]:


final_preds = get_predictions(test_predictions_1, test_predictions_2, test_predictions_3)


# In[74]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_y, final_preds)
cm


# In[75]:


import itertools

def plot_confusion_matrix_modified(cm, target_names, title='Confusion Matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues') # With summer few values aren't visible

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[76]:


plot_confusion_matrix_modified(cm, ["0", "1", "2", "3", "4"], normalize=False)


# In[77]:


plot_confusion_matrix_modified(cm, ["0", "1", "2", "3", "4"], normalize=True)


# In[ ]:





# In[ ]:




