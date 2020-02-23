#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf


# In[ ]:


data


# In[66]:


import glob
data = pd.concat([pd.read_csv(f) for f in glob.glob('se*.csv')])
print(data.columns)
data.dropna(inplace=True)
target = data.pop('flowrate')
m = np.median(target)
v = np.var(target)**(.5)
#target = target-np.min(target)
#target = (target/v)
#target = target*(100/np.max(target))
print(data)
data.pop('Unnamed: 0')
batch_size = 256
training_set_size = int(.2*len(data))
dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values)).shuffle(len(data)).batch(batch_size, drop_remainder=True)
testset = dataset.take(training_set_size//batch_size)
trainset = dataset.skip(training_set_size//batch_size).repeat(10)
np.any(np.isnan(target))
len(data)


# In[ ]:


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import time


# In[ ]:


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


# In[42]:


import keras
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, PReLU


# In[68]:


middle_layer_size = 64
model = Sequential([
Dense(middle_layer_size, input_shape=(7,)),
PReLU(),
Dense(middle_layer_size),
PReLU(),
Dense(middle_layer_size),
PReLU(),
Dense(middle_layer_size),
PReLU(),
Dense(middle_layer_size),
Dense(1)])

model.compile(optimizer=tf.optimizers.Adam(.01),
            loss='mse',
            metrics=['mape'])


# In[ ]:


''


# In[69]:



history = model.fit(trainset, validation_data=testset,epochs=200)
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


tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)


# In[ ]:


import glob
for f in [f for f in glob.glob('*.csv')]:
    print(f)
    data = pd.read_csv(f)
    data.dropna(inplace=True)
    data.pop('Unnamed: 0')
    target = data.pop('flowrate')
    batch_size = 128
    dataset = tf.data.Dataset.from_tensor_slices((data.values, target.values)).shuffle(len(data)).batch(batch_size, drop_remainder=True)
    model.evaluate(dataset)


# In[ ]:


model.summary()


# In[55]:


for layer in model.layers:
    w = layer.get_weights()
    print(w)


# In[ ]:




