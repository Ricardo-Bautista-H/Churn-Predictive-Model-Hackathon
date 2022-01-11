#!/usr/bin/env python
# coding: utf-8

# # Data Scientist Hackathon

# ### Ricardo Bautista Huerta

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sb
import math
import warnings


# In[2]:


warnings.filterwarnings("ignore")
style.use('ggplot')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# ## Cargando bases de datos

# In[3]:


clients = pd.read_csv('clients_table.txt', sep=",")
clients


# In[4]:


credit_score = pd.read_csv('credit_score_table.txt', sep=",")
credit_score


# In[5]:


products = pd.read_csv('products_table.txt', sep=",")
products


# In[6]:


transactions = pd.read_csv('transactions_table.txt', sep=",")
transactions


# ## Filtrando

# ### Contratos del 2015 en adelante

# In[7]:


clients['application_date'] = pd.to_datetime(clients['application_date'], format='%Y/%m/%d')
clients


# In[8]:


clients['application_date'].dt.year.unique()


# In[9]:


clients = clients[clients['application_date'].dt.year >= 2015]
clients


# In[10]:


clients['application_date'].dt.year.unique()


# In[11]:


clients.shape


# ### Las operaciones estuvieron cerradas en el 2019 en Italia

# In[12]:


clients['Geography'].unique()


# In[13]:


clients['exit_date'] = pd.to_datetime(clients['exit_date'], format='%Y/%m/%d')
clients


# In[14]:


clients['application_date'].dt.year.unique()


# In[15]:


clients = clients.drop(clients[(clients['Geography'] == 'Italy') & (clients['application_date'].dt.year == 2019)].index)
clients


# In[16]:


clients.shape


# ### Un contrato

# In[17]:


clients[clients.duplicated(subset=['CustomerId'],keep=False)].sort_values(by=['CustomerId'])


# In[18]:


clients = clients.drop_duplicates(subset=['CustomerId'])
clients


# In[19]:


clients.shape


# ### Información Perdida

# In[20]:


clients['Missing'] = clients.isnull().sum(axis=1)
clients


# In[21]:


clients.isnull().sum()


# In[22]:


criterio = (len(clients.columns) - 1) * 0.75 #75% de información perdida
criterio


# In[23]:


clients['Missing'].max()


# In[24]:


clients.drop(clients[clients.Missing > criterio].index, inplace = True)


# In[25]:


clients.drop(columns='Missing',inplace=True)


# In[26]:


clients


# In[27]:


clients.shape


# ### 2 años de información

# In[28]:


clients['Años_Info'] = clients['application_date'].dt.year
clients['Años_Info'] = 2019 - clients['Años_Info']
clients


# In[29]:


clients['Años_Info'].unique()


# In[30]:


clients.drop(clients[clients.Años_Info >= 2].index, inplace=True)


# In[31]:


clients.drop(columns='Años_Info',inplace=True)


# In[32]:


clients


# In[33]:


clients.shape


# ## Otras variables relevantes

# ### Número de productos

# In[34]:


products_group_ = products.groupby(['CustomerId']).agg({'Products':'count'})
products_group = products_group_.reset_index()
products_group


# In[35]:


clients = pd.merge(clients,products_group,on='CustomerId',how='left')
clients


# ### Balance Account

# In[36]:


transactions.sort_values(['CustomerId', 'Transaction'])


# In[37]:


transactions.columns


# In[38]:


balance = transactions.groupby(['CustomerId']).agg({'Value':'sum'}).reset_index()
balance


# In[39]:


clients = pd.merge(clients,balance,on='CustomerId',how='left')
clients


# In[40]:


clients.rename(columns={"Value": "Balance"}, inplace=True)


# ### Credit Score

# In[41]:


credit_score['Date'] = pd.to_datetime(credit_score['Date'], format='%Y/%m')
credit_score.sort_values(['CustomerId', 'Date'])


# In[42]:


credit_score['Month'] = credit_score['Date'].dt.month
credit_score['Year'] = credit_score['Date'].dt.year
credit_score


# In[43]:


clients['Month'] = clients['application_date'].dt.month
clients['Year'] = clients['application_date'].dt.year
clients


# In[44]:


clients = pd.merge(clients, credit_score, on=['CustomerId', 'Month','Year'])
clients.drop(['Month', 'Year','Date'], axis = 1 , inplace = True)
clients


# ### Age

# In[45]:


clients


# In[46]:


clients['birth_date'] = pd.to_datetime(clients['birth_date'], format='%Y/%m/%d')
clients['Age'] = (clients['application_date'] - clients['birth_date']) / np.timedelta64(1, 'Y')
clients['Age'] = clients['Age'].apply(np.floor)
clients


# In[47]:


clients.shape


# 491796 Registros

# In[48]:


clients.describe()


# ## Modelado

# In[49]:


from collections import Counter
from sklearn import preprocessing
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import plot_tree


# In[50]:


style.use('ggplot')


# ### Creando churn

# In[51]:


clients['churn'] = np.where(clients['exit_date'].isna(), 0, 1)
clients


# ### Drop CustomerId

# In[52]:


clients.drop('CustomerId',axis='columns',inplace=True)


# In[53]:


clients


# ### Eliminando missing values en Gender

# In[54]:


clients.isnull().sum()


# In[55]:


clients = clients[clients['Gender'].notna()]
# Dropeo los na gender porque es una variable no imputable y debe ser incluida en el modelo


# In[56]:


clients.isnull().sum()
# Los otros missing values también se eliminaron


# In[57]:


clients['Age'] = clients['Age'].astype(np.int64)
clients['HasCrCard'] = clients['HasCrCard'].astype(np.int64)
clients['IsActiveMember'] = clients['IsActiveMember'].astype(np.int64)


# In[58]:


clients.shape


# ### Generar categórica Gender

# In[59]:


label = preprocessing.LabelEncoder() 
 
clients['Gender']= label.fit_transform(clients['Gender']) # 0 Female - 1 Male
clients


# In[60]:


num_bins = 10
clients.hist(bins = num_bins, figsize=(20,10))
plt.show()


# ### Dividir Test y Train

# In[61]:


X = clients[['Gender','HasCrCard','IsActiveMember', 'EstimatedSalary','Products','Balance','Score','Age']]
Y = clients.iloc[:,13]


# In[62]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# ### Probando Modelos

# In[63]:


models = ["Nearest_Neighbors", "Decision_Tree", "Random_Forest", "Neural_Net"]


# In[64]:


classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000)]
# Usando Hiperparámetros default


# In[181]:


scores = []
for model, clf in zip(models, classifiers):
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    scores.append(score)
# Comparamos accuracy de los clasificadores con la línea base


# In[182]:


scores


# In[184]:


sc = pd.DataFrame()
sc['models'] = models
sc['score'] = scores
sc


# Entre los tres mejores clasificadores base destacan el Decision Tree, Random Forest y la Red Neuronal. Como la mejora de accuracy respecto al primero no es muy significativa, se usará el Decision Tree al permitirnos tener resultados más interpetables y sencillos que los otros dos, además que resulta en menores tiempos de procesamiento.
# 
# Por otra parte, la base de datos posee 484527 registros y se utilizarán 8 variables para el modelo, por lo que la data sí es suficientemente grande para usar este algoritmo.

# ### Elaborando Árbol de decisiones y oversampling

# In[65]:


def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f1: {}".format(f1_score(true_value, pred)))


# In[66]:


classifier = DecisionTreeClassifier


# In[67]:


pipeline = make_pipeline(classifier(random_state=42, max_depth=3))
model = pipeline.fit(X_train, Y_train)
prediction = model.predict(X_test)


# Es necesario notar que existe un problema de desbalanceo: tanto en el train como el el test de la data, casi el 80% corresponde a que no es churn (no ha cancelado el producto dentro de los dos años)

# In[68]:


Y_train.value_counts()


# In[69]:


Y_train.value_counts(normalize=True)*100


# In[70]:


Y_test.value_counts()


# In[71]:


Y_test.value_counts(normalize=True)*100


# Para solucionar el problema se utilizará SMOTE imblearn para balancear por oversampling y mejorar las predicciones del modelo.

# In[72]:


smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), classifier(max_depth=3, random_state=42))
smote_model = smote_pipeline.fit(X_train, Y_train)
smote_prediction = smote_model.predict(X_test)


# In[73]:


print()
print("normal data distribution: {}".format(Counter(Y)))
X_smote, Y_smote = SMOTE().fit_resample(X, Y)
print("SMOTE data distribution: {}".format(Counter(Y_smote)))


# In[74]:


print(classification_report(Y_test, prediction))
print(classification_report_imbalanced(Y_test, smote_prediction))


# In[75]:


print()
print_results("normal classification", Y_test, prediction)
print()
print_results("SMOTE classification", Y_test, smote_prediction)


# ### Gráficos

# In[76]:


feature_names = list(X.columns.values)
Class_names = ['Not_Churn', 'Churn']


# In[77]:


# Baseline
fig = plt.figure(figsize=(25,20))
plot_tree(model['decisiontreeclassifier'],feature_names= feature_names, class_names= Class_names,filled=True)


# In[78]:


# Final
fig = plt.figure(figsize=(25,20))
plot_tree(smote_model['decisiontreeclassifier'],feature_names= feature_names,
          class_names= Class_names, filled=True, rounded=True)

