#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[2]:


data


# In[19]:


import glob
data = pd.concat([pd.read_csv(f) for f in glob.glob('*.csv')])
data.dropna(inplace=True)
target = data.pop('flowrate')
#m = np.mean(target)
#v = np.var(target)
#target = target-m
#target = target/m
print(np.min(target))
data.pop('Unnamed: 0')
batch_size = 64
training_set_size = int(.2*len(data))
dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values)).shuffle(len(data)).batch(batch_size, drop_remainder=True)
testset = dataset.take(training_set_size//batch_size)
trainset = dataset.skip(training_set_size//batch_size)
np.any(np.isnan(target))
len(data)


# In[4]:


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import time


# In[5]:


def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Use of this metric is not recommended; for illustration only. 
    See other regression metrics on sklearn docs:
      http://scikit-learn.org/stable/modules/classes.html#regression-metrics
    Use like any other metric
    >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    Out[]: 24.791666666666668
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


start_time = time.time()
svm_clf = Pipeline([
    ("scaler", StandardScaler()), 
    ("linear_svc", SVR(kernel='rbf', gamma='scale', coef0=.01, C=200))
])
a_train, a_test, b_train, b_test = train_test_split(data, target, test_size=0.2)
svm_clf.fit(a_train, b_train)
print(time.time()-start_time)
from sklearn.metrics import mean_absolute_error
print(mean_absolute_percentage_error(b_test, svm_clf.predict(a_test)))


# In[4]:


np.any(np.isnan(target))


# In[22]:


model = tf.keras.Sequential([
tf.keras.layers.Dense(500),
tf.keras.layers.Dense(500),
tf.keras.layers.Dense(500),
tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam',
            loss='mse',
            metrics=['mape'])


# In[9]:


model.summary()


# In[23]:


history = model.fit(trainset, validation_data=testset,epochs=100)
# Visualize training history
import matplotlib.pyplot as plt
import numpy
get_ipython().run_line_magic('matplotlib', 'notebook')
# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['mape'])
plt.plot(history.history['val_mape'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# for pred, true in zip(model.predict(data), target):
#     print(pred[0]-true)

# In[ ]:


len(train)


# In[11]:


model.summary()


# In[ ]:




