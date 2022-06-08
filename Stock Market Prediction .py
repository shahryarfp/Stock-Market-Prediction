#!/usr/bin/env python
# coding: utf-8

# # Q1

# In[142]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU
import datetime
import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import RMSprop


# ## Reading Dataset

# In[143]:


AAPL = pd.read_csv('AAPL.csv', sep='\t')
GOOG = pd.read_csv('GOOG.csv', sep='\t')


# In[144]:


AAPL


# In[145]:


GOOG


# ## Preprocessing

# In[146]:


AAPL_y = AAPL['Close']
GOOG_y = GOOG['Close']
AAPL.drop('Date', axis=1, inplace=True)
GOOG.drop('Date', axis=1, inplace=True)
AAPL.drop('Close', axis=1, inplace=True)
GOOG.drop('Close', axis=1, inplace=True)

#Normalizing
AAPL = (AAPL-AAPL.min())/(AAPL.max()-AAPL.min())
GOOG = (GOOG-GOOG.min())/(GOOG.max()-GOOG.min())
AAPL_y = (AAPL_y-AAPL_y.min())/(AAPL_y.max()-AAPL_y.min())
GOOG_y = (GOOG_y-GOOG_y.min())/(GOOG_y.max()-GOOG_y.min())


# In[147]:


num_of_rows = len(AAPL)
time_series_length = 30
inputs = []
outputs = []
for i in range(num_of_rows-time_series_length-1):
  new_input = []
  new_output = []
  for j in range(time_series_length):
    new_input_data = []
    lst = AAPL.columns.values
    for title in lst:
      new_input_data.append(AAPL[title][i+j])
    lst = GOOG.columns.values
    for title in lst:
      new_input_data.append(GOOG[title][i+j])
    new_input.append(new_input_data)

    if j == 29:
      new_output = []
      new_output.append(AAPL_y[i+j+1])
      new_output.append(GOOG_y[i+j+1])

  inputs.append(new_input)
  outputs.append(new_output)

inputs = np.array(inputs)
print('Inputs Shape:', inputs.shape)
outputs = np.array(outputs)
print('Outputs Shape:', outputs.shape)


# ##### Splitting data

# In[148]:


test_x = inputs[int(0.85*outputs.shape[0]):]
test_y = outputs[int(0.85*outputs.shape[0]):]
train_x = inputs[:int(0.85*outputs.shape[0])]
train_y = outputs[:int(0.85*outputs.shape[0])]


# ## Creating Model, Fitting and Getting Results

# ### LSTM

# In[149]:


model = Sequential()
model.add(LSTM(64, input_shape = (time_series_length, inputs.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(2, activation = 'linear'))

my_optimizer = 'RMSprop'
my_loss = 'mse'
model.compile(loss = my_loss, optimizer = my_optimizer, metrics = ['mse','mae'])
model.summary()


# In[150]:


start = datetime.datetime.now()
trainedModel = model.fit(train_x, train_y, batch_size = 64, epochs = 60, validation_split = 0.2)
end = datetime.datetime.now()


# In[151]:


print('Training Duration: ', end-start)


# In[152]:


history = trainedModel.history
mae = history['mae']
val_mae = history['val_mae']
mse = history['mse']
val_mse = history['val_mse']

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.plot(mae)
plt.plot(val_mae)
plt.legend(['mae','val_mae'])
plt.figure()

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.plot(mse)
plt.plot(val_mse)
plt.legend(['mse','val_mse'])


# In[153]:


#plotting predicted vs real
pred_y = model.predict(test_x)
plt.title('AAPL')
plt.xlabel('y_test')
plt.ylabel('y_predicted')
for i in range(len(pred_y)):
  plt.scatter(test_y[i][0], pred_y[i][0])
plt.show()

plt.title('GOOG')
plt.xlabel('y_test')
plt.ylabel('y_predicted')
for i in range(len(pred_y)):
  plt.scatter(test_y[i][1], pred_y[i][1])


# ### RNN

# In[154]:


model = Sequential()
model.add(SimpleRNN(64, input_shape = (time_series_length, inputs.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(2, activation = 'linear'))

model.compile(loss = my_loss, optimizer = my_optimizer, metrics = ['mse','mae'])
model.summary()


# In[155]:


start = datetime.datetime.now()
trainedModel = model.fit(train_x, train_y, batch_size = 64, epochs = 60, validation_split = 0.2)
end = datetime.datetime.now()


# In[156]:


print('Training Duration: ', end-start)


# In[157]:


history = trainedModel.history
mae = history['mae']
val_mae = history['val_mae']
mse = history['mse']
val_mse = history['val_mse']

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.plot(mae)
plt.plot(val_mae)
plt.legend(['mae','val_mae'])
plt.figure()

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.plot(mse)
plt.plot(val_mse)
plt.legend(['mse','val_mse'])


# In[158]:


#plotting predicted vs real
pred_y = model.predict(test_x)
plt.title('AAPL')
plt.xlabel('y_test')
plt.ylabel('y_predicted')
for i in range(len(pred_y)):
  plt.scatter(test_y[i][0], pred_y[i][0])
plt.show()

plt.title('GOOG')
plt.xlabel('y_test')
plt.ylabel('y_predicted')
for i in range(len(pred_y)):
  plt.scatter(test_y[i][1], pred_y[i][1])


# ### GRU

# In[159]:


model = Sequential()
model.add(GRU(64, input_shape = (time_series_length, inputs.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(2, activation = 'linear'))

model.compile(loss = my_loss, optimizer = my_optimizer, metrics = ['mse','mae'])
model.summary()


# In[160]:


start = datetime.datetime.now()
trainedModel = model.fit(train_x, train_y, batch_size = 64, epochs = 60, validation_split = 0.2)
end = datetime.datetime.now()


# In[161]:


print('Training Duration: ', end-start)


# In[162]:


history = trainedModel.history
mae = history['mae']
val_mae = history['val_mae']
mse = history['mse']
val_mse = history['val_mse']

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.plot(mae)
plt.plot(val_mae)
plt.legend(['mae','val_mae'])
plt.figure()

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.plot(mse)
plt.plot(val_mse)
plt.legend(['mse','val_mse'])


# In[163]:


#plotting predicted vs real
pred_y = model.predict(test_x)
plt.title('AAPL')
plt.xlabel('y_test')
plt.ylabel('y_predicted')
for i in range(len(pred_y)):
  plt.scatter(test_y[i][0], pred_y[i][0])
plt.show()

plt.title('GOOG')
plt.xlabel('y_test')
plt.ylabel('y_predicted')
for i in range(len(pred_y)):
  plt.scatter(test_y[i][1], pred_y[i][1])


# In[163]:





# In[163]:





# In[163]:





# In[163]:





# In[163]:





# In[163]:





# In[163]:




