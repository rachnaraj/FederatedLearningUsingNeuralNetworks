# coding: utf-8

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')

# We only need to use it if we are taking inputs from google colab


# ### Libraries

# In[ ]:


from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import tensorflow as tf
# import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD # from keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop # instead of from keras.optimizers import RMSprop
# from keras import datasets

# from keras.callbacks import LearningRateScheduler
# from keras.callbacks import History

from tensorflow.keras import losses
from sklearn.utils import shuffle


# for saving model
from keras.models import model_from_json
import h5py

print(tf.__version__)
print(tf.keras.__version__)


import sys
# sys.argv[0] # prints python_script.py
storage_loc =  sys.argv[1]

# ### taking input

# In[ ]:

K = 25 # as we have 25 clients

clients = {} # dictionary
clients_cut = {}

## reading the data from each client
for st_code in range(101,126):
    # clients[st_code] = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/dataNew/station_code"+str(st_code)+".csv", index_col = 0)
    clients[st_code] = pd.read_csv(storage_loc+"/dataNew/station_code"+str(st_code)+".csv", index_col = 0) #for running on server

    clients[st_code]['Measurement date'] = pd.to_datetime(clients[st_code]['Measurement date'])

    clients[st_code]['year'] = clients[st_code]['Measurement date'].dt.year
    clients[st_code]['month'] = clients[st_code]['Measurement date'].dt.month
    clients[st_code]['week'] = clients[st_code]['Measurement date'].dt.week
    clients[st_code]['day'] = clients[st_code]['Measurement date'].dt.day
    clients[st_code]['hour'] = clients[st_code]['Measurement date'].dt.hour
    clients[st_code]['minute'] = clients[st_code]['Measurement date'].dt.minute # minute is not significant; as only 0 values
    clients[st_code]['dayOfWeek'] = clients[st_code]['Measurement date'].dt.dayofweek

    # choosing features
    clients[st_code] = clients[st_code][['Latitude', 'Longitude','year', 'month', 'week', 'day', 'hour', 'dayOfWeek', 'CO', 	'PM10',	'PM2.5']]

    clients[st_code].drop(clients[st_code][clients[st_code]['CO'] < 0].index, axis=0, inplace=True)
    clients[st_code].drop(clients[st_code][clients[st_code]['PM10'] < 0].index, axis=0, inplace=True)
    clients[st_code].drop(clients[st_code][clients[st_code]['PM2.5'] < 0].index, axis=0, inplace=True)


# complete dataset
frames = list(clients.values())
dataset = pd.concat(frames)
# display(dataset)


# ## 4 hidden layer
# * with 64 neurons

# ### traditional Learning

# In[ ]:


def traditional_ML_4L_64_train(loss_p, gas, train_data):
  # initialize w_0
  model = Sequential()

  # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(1, kernel_initializer= 'he_uniform', activation='linear')) # output layer


  # compile the model
  model.compile(loss='mse', 
              optimizer='adam', 
              metrics=['mse'])
  
  x_train, y_train = train_data[['Latitude', 'Longitude','year', 'month', 'week', 'day', 'hour', 'dayOfWeek']],  train_data[gas]

 

  B = 1000 #int(len(x_train) * 0.01) # so for each client the batch size would be different depending upon the total sample size
  E = 25 #max(30, B//10) # number of local epochs; it will also depend on sample size (indirectly)


  model.fit(x_train, y_train, 
                  epochs = E, 
                  batch_size=B,
                  verbose=False)
  
  return model
      


# In[ ]:


def traditional_ML_4L_64_test(model_name, gas, test_data):

  ## loading the json and creating model
  json_file = open(model_name + ".json", 'r')
  model_json = json_file.read()
  json_file.close()

  model = model_from_json(model_json)
  model.load_weights(model_name + ".h5")

  ## need to compile the model again after loading
  # compile the model
  model.compile(loss='mse', 
              optimizer='adam', 
              metrics=['mse'])
  

  x_test, y_test = test_data[['Latitude', 'Longitude','year', 'month', 'week', 'day', 'hour', 'dayOfWeek']],  test_data[gas]

  x_test, y_test = x_test.to_numpy(), y_test.to_numpy()

  # adjusting the dimensions
  x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
  y_test = y_test.reshape(y_test.shape[0],1)
  
  y_predicted = model.predict(x_test)
  

#   from sklearn.metrics import mean_squared_error
#   mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())

  NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))
  return NE


# In[ ]:


# Final plotting
K = 25 # as we have 25 clients
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gases = ['CO', 	'PM10',	'PM2.5']
errors_tl_h4_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
models = []
model_name = "TL_model_4L_64_"


for gas in gases:
  for lp in loss_prob:
      for st_code in range(101,126):
          # splitting into train test split
          X, y = clients[st_code][['Latitude', 'Longitude','year', 'month', 'week', 'day', 'hour', 'dayOfWeek']],  clients[st_code][gas]
          x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = lp, random_state=1)

          # Normalization using min-max scaler
          # Normalization across instances should be done after splitting the data between training and test set, using only the data from the training set.
          # to avoid data leak
          scaler_cols = x_train.columns
          scaler_idx_train = x_train.index
          scaler_idx_test = x_test.index

          # from sklearn.preprocessing import MinMaxScaler
          scaler = MinMaxScaler()

          # transforming train data
          x_train_encoded_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns = scaler_cols, index=scaler_idx_train)
          # display(x_train_encoded_scaled.head())

          # transforming test data
          x_test_encoded_scaled = pd.DataFrame(scaler.transform(x_test), columns = scaler_cols, index = scaler_idx_test)
          # display(x_test_encoded_scaled.head())

          # training data prep
          if st_code == 101:
            train_data = pd.concat([x_train_encoded_scaled, y_train], axis = 1)
          else:
            xy = pd.concat([x_train_encoded_scaled, y_train], axis=1)
            train_data = pd.concat([train_data, xy], axis=0)
          # clients_cut[st_code] = pd.concat([x_train_encoded_scaled, y_train], axis = 1)
          

          # testing data prep
          # first appending x_test and y_test column wise
          # then appending each clients' dataframe one below other
          if st_code == 101:
            test_data = pd.concat([x_test_encoded_scaled, y_test], axis = 1)
          else:
            xy = pd.concat([x_test_encoded_scaled, y_test], axis=1)
            test_data = pd.concat([test_data, xy], axis=0)
          # x_test, y_test = x_test.to_numpy(), y_test.to_numpy()
      
      # Training the model using traditional learning
      model = traditional_ML_4L_64_train(lp, gas, train_data) #### COMMENT THIS LINE ONCE TRAINING AND SAVING OF MODEL IS DONE

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_" 


      #### COMMENT THIESE 3 LINES ONCE TRAINING AND SAVING OF MODEL IS DONE
      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # # serialize weights to HDF5
      model.save_weights(model_address + ".h5") #### COMMENT THIS LINE ONCE TRAINING AND SAVING OF MODEL IS DONE


      ## Testing model
      error = traditional_ML_4L_64_test(model_address, gas, train_data)
      errors_tl_h4_64[gas].append(error)


# ### Federated Learning

# In[ ]:


# ClientUpdate(k, w): // Run on client k
def Client_L4_64(client_idx, model, gas):
    # B ← (split P_k into batches of size B) # what is p_K --> each clients' sample size
    B = 1000 #int(len(clients[client_idx]) * 0.01) # so for each client the batch size would be different depending upon the total sample size
    E = 25 #max(30, B//10) # number of local epochs; it will also depend on sample size (indirectly)

    data = clients_cut[client_idx]
    # data = dataset.loc[dataset['Station code'] == client_idx]

    X, y = data[['Latitude', 'Longitude','year', 'month', 'week', 'day', 'hour', 'dayOfWeek']],  data[gas]

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=1)

    try:
        model.fit(x_train, y_train, 
                    epochs = E, 
                    batch_size=B,
                    validation_data=(x_val, y_val),
                    verbose=False)
    except Exception as e:
        print(e)
        print("error")

    # return w to server
    # in our case no need for any explicit return; as the model is pass by value
    return data.shape[0] # this will return the number of total sample


# In[ ]:


# Server executes: 
def Server_L4_64(gas, T):
    # initialize w_0
    model = Sequential()
    
    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(1, kernel_initializer= 'he_uniform', activation='linear')) # output layer


    # compile the model
    model.compile(loss='mse', 
                optimizer='adam', 
                metrics=['mse'])
    

# for each round t = 1, 2, . . . do
    for t in range(1, T):
        C = np.random.random(1)[0] # random number between 0 and 1.

        # m ← max(C  K, 1)
        m = max(int(C*K), 1) # m == random number of client selected
        weight_t_plus_1 = [None] * m # matrix to store the weights of all client in each t-round
        n_k = [None] * m # parameters for weighted sum 


        # S_t ← (random set of m clients)
        # S = {} # dictionary
        S_t = np.random.uniform(low=101, high=126, size=(m)).astype(int)

        # No need for below loop, as server don't need access to client data, server only need clients' number
        # for client in m_clients:
        #   S[client] = clients[client]

        initial_weights = model.get_weights() # setting initial weights; should be same for all clients
        # for t= 1 to T, initial_weights would be t-1th's final weights

        # for each client k ∈ S_t in parallel do
        client_idx = 0
        for client in S_t:
            # w^(k)_(t+1) ← ClientUpdate(k, model)
            n_k[client_idx] = Client_L4_64(client, model, gas) # pass by reference for model
            weight_t_plus_1[client_idx] = model.get_weights()
            client_idx += 1

            # setting weights back to initial weights
            model.set_weights(initial_weights)

        # finding the weighted sum
        final_weights_t = np.array(weight_t_plus_1[0]) * (n_k[0] / sum(n_k))

        for idx in range(1, m):
          # w_(t+1) ← summation(k=1 to K){(n_k/n) * w^(k)_(t+1)} #n_k - no. of training sample in each client K; n - total training samples;
          final_weights_t += np.array(weight_t_plus_1[idx]) * (n_k[idx]/ sum(n_k))

        # setting the aggregated weights
        model.set_weights(final_weights_t)


#         print(model.get_weights())
    # return error_clients(model, loss_p)) # THIS LINE WON'T BE THERE IN REAL SERVER
    return model


# In[ ]:


def error_clients_L4_64(model_name, gas, test_data):

    ## loading the json and creating model
    json_file = open(model_name + ".json", 'r')
    model_json = json_file.read()
    json_file.close()

    model = model_from_json(model_json)
    model.load_weights(model_name + ".h5")

    ## need to compile the model again after loading
    # compile the model
    model.compile(loss='mse', 
                optimizer='adam', 
                metrics=['mse'])
  
    
    X, y = test_data[['Latitude', 'Longitude','year', 'month', 'week', 'day', 'hour', 'dayOfWeek']],  test_data[gas]
    
    # adjusting the dimensions
    x_test, y_test = X.to_numpy(), y.to_numpy()
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
    y_test = y_test.reshape(y_test.shape[0],1)

    y_predicted = model.predict(x_test)
    
    from sklearn.metrics import mean_squared_error
    # print("y_predicted", y_predicted)
    # print("y_predicted.ravel()", y_predicted.ravel())
    NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))

    # mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())
    
    return NE

          


# In[ ]:



# create an Empty DataFrame object
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
T_round = [15, 20, 25, 35, 50, 75, 100]
models = []
errors_fl_h4_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
#errors_tl_h4_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
gases = ['CO', 	'PM10',	'PM2.5']
model_name = "FL_model_4L_64_"


# reading all the clients and building model
for gas in gases:
  for lp in loss_prob:
      for st_code in range(101,126):
          # splitting into train test split
          X, y = clients[st_code][['Latitude', 'Longitude','year', 'month', 'week', 'day', 'hour', 'dayOfWeek']],  clients[st_code][gas]
          x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = lp, random_state=1)

          # Normalization using min-max scaler
          # Normalization across instances should be done after splitting the data between training and test set, using only the data from the training set.
          # to avoid data leak
          scaler_cols = x_train.columns
          scaler_idx_train = x_train.index
          scaler_idx_test = x_test.index

          # from sklearn.preprocessing import MinMaxScaler
          scaler = MinMaxScaler()

          # transforming train data
          x_train_encoded_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns = scaler_cols, index=scaler_idx_train)
          # display(x_train_encoded_scaled.head())

          # transforming test data
          x_test_encoded_scaled = pd.DataFrame(scaler.transform(x_test), columns = scaler_cols, index = scaler_idx_test)
          # display(x_test_encoded_scaled.head())

          # training data prep
          clients_cut[st_code] = pd.concat([x_train_encoded_scaled, y_train], axis = 1)

          # testing data prep
          # first appending x_test and y_test column wise
          # then appending each clients' dataframe one below other
          if st_code == 101:
            test_data = pd.concat([x_test_encoded_scaled, y_test], axis = 1)
          else:
            xy = pd.concat([x_test_encoded_scaled, y_test], axis=1)
            test_data = pd.concat([test_data, xy], axis=0)
          # x_test, y_test = x_test.to_numpy(), y_test.to_numpy()

      error = [] # empty array for each loss prob
      
      for T in T_round:
        # Training the model using federated learning
        model = Server_L4_64(gas,T) ## COMMENT IT ONCE THE MODEL TRAINING IS DONE

        ## Saving model
        ## serializing model to JSON
        model_address = storage_loc +"/" + model_name + gas + str(lp) + "_" + str(T) + "_" 


        ## COMMENT BELOW LINES ONCE THE MODEL IS SAVED
        model_json = model.to_json()
        with open(model_address + ".json", "w") as json_file:
          json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(model_address + ".h5") ## COMMENT THIS LINE ONCE THE MODEL IS SAVED


        ## Testing model
        error.append(error_clients_L4_64(model_address, gas, test_data))

      # append the errors for each loss prob
      errors_fl_h4_64[gas].append(error)


# ### Plots

# #### Original plots

# In[ ]:
## TEST DATA SAMPLES for Error
# errors_fl_h4_64 = {'CO':[[0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56,0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71]], 	
# 'PM10':[[0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56,0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71]],	
# 'PM2.5':[[0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56,0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71], [0.12, 0.34, 0.56, 0.78, 0.76, 0.79, 0.71]]}
# loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# errors_tl_h4_64 = {'CO':[0.12,0.34, 0.56, 0.79, 0.76,0.78,0.71,0.76,0.77], 	
# 'PM10':[0.12,0.34, 0.56, 0.79, 0.76,0.78,0.71,0.76,0.77],	
# 'PM2.5':[0.12,0.34, 0.56, 0.79, 0.76,0.78,0.71,0.76,0.77]}

error_fl_CO = np.array(errors_fl_h4_64['CO']).T
error_fl_PM25 = np.array(errors_fl_h4_64['PM2.5']).T
error_fl_PM10 = np.array(errors_fl_h4_64['PM10']).T


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
plt.figure(figsize=(12, 8))



ax = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[0], markers=True,
                  marker="o", dashes=False, label = 'FL t:= 15', linewidth=2,ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[1], markers=True,
                  marker="s", dashes=False, label = 'FL t:= 20', linewidth=2,ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[2], markers=True,
                  marker="P", dashes=False, label = 'FL t:= 25', linewidth=2,ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[3], markers=True,
                  marker="X", dashes=False, label = 'FL t:= 35', linewidth=2,ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[4], markers=True,
                  marker="v", dashes=False, label = 'FL t:= 50', linewidth=2,ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[5], markers=True,
                  marker="D", dashes=False, label = 'FL t:= 75', linewidth=2,ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[6], markers=True,
                  marker="<", dashes=False, label = 'FL t:= 100', linewidth=2,ms='10', mew='2')




# ax.set(ylim=(0.2, 0.5))
# ax.set(ylim=(0.0, 1.0))
ax.grid(True)
plt.legend(fontsize=11)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE Error",fontsize=14)
plt.title("PM2.5 [Architechture - 4HL/64Neuron]", fontsize=14) 
plt.savefig(storage_loc+"/PM2.5 [Architechture - 4HL-64Neuron] for various t-values.png", format="png")
# plt.show(ax)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
plt.figure(figsize=(12, 8))



ax1 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[0], markers=True,
                  marker="o", dashes=False, label = 'FL t:= 15', linewidth=2,ms='10', mew='2')


ax1 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[1], markers=True,
                  marker="s", dashes=False, label = 'FL t:= 20', linewidth=2,ms='10', mew='2')


ax1 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[2], markers=True,
                  marker="P", dashes=False, label = 'FL t:= 25', linewidth=2,ms='10', mew='2')


ax1 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[3], markers=True,
                  marker="X", dashes=False, label = 'FL t:= 35', linewidth=2,ms='10', mew='2')


ax1 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[4], markers=True,
                  marker="v", dashes=False, label = 'FL t:= 50', linewidth=2,ms='10', mew='2')


ax1 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[5], markers=True,
                  marker="D", dashes=False, label = 'FL t:= 75', linewidth=2,ms='10', mew='2')


ax1 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[6], markers=True,
                  marker="<", dashes=False, label = 'FL t:= 100', linewidth=2,ms='10', mew='2')




# ax.set(ylim=(0.2, 0.5))
# ax.set(ylim=(0.0, 1.0))
ax1.grid(True)
plt.legend(fontsize=11)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE Error",fontsize=14)
plt.title("CO [Architechture - 4HL/64Neuron]") 
# plt.show(ax1)
plt.savefig(storage_loc+"/CO [Architechture - 4HL-64Neuron] for various t-values.png", format="png")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
plt.figure(figsize=(12, 8))



ax2 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[0], markers=True,
                  marker="o", dashes=False, label = 'FL t:= 15', linewidth=2,ms='10', mew='2')


ax2 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[1], markers=True,
                  marker="s", dashes=False, label = 'FL t:= 20', linewidth=2,ms='10', mew='2')


ax2 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[2], markers=True,
                  marker="P", dashes=False, label = 'FL t:= 25', linewidth=2,ms='10', mew='2')


ax2 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[3], markers=True,
                  marker="X", dashes=False, label = 'FL t:= 35', linewidth=2,ms='10', mew='2')


ax2 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[4], markers=True,
                  marker="v", dashes=False, label = 'FL t:= 50', linewidth=2,ms='10', mew='2')


ax2 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[5], markers=True,
                  marker="D", dashes=False, label = 'FL t:= 75', linewidth=2,ms='10', mew='2')


ax2 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[6], markers=True,
                  marker="<", dashes=False, label = 'FL t:= 100', linewidth=2,ms='10', mew='2')




# ax.set(ylim=(0.2, 0.5))
# ax.set(ylim=(0.0, 1.0))
ax2.grid(True)
plt.legend(fontsize=11)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE Error",fontsize=14)
plt.title("PM10 [Architechture - 4HL/64Neuron]") 
# plt.show(ax2)
plt.savefig(storage_loc+"/PM10 [Architechture - 4HL-64Neuron] for various t-values.png", format="png")


# #### FL vs TL

# In[ ]:


plt.figure(figsize=(12, 8))

# plt.subplot(3,1,1)
ax3 = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['CO'])[:], markers=True, 
                  marker="D", dashes=False, label = 'TL Error', linewidth=4, mfc='red', ms='8', mew='2')

ax3 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[0], markers=True,
                  marker="o", dashes=False, label = 'FL t:= 15', linewidth=2,ms='10', mew='2')


ax3 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[1], markers=True,
                  marker="s", dashes=False, label = 'FL t:= 20', linewidth=2,ms='10', mew='2')


ax3 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[2], markers=True,
                  marker="P", dashes=False, label = 'FL t:= 25', linewidth=2,ms='10', mew='2')


ax3 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[3], markers=True,
                  marker="X", dashes=False, label = 'FL t:= 35', linewidth=2,ms='10', mew='2')


ax3 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[4], markers=True,
                  marker="v", dashes=False, label = 'FL t:= 50', linewidth=2,ms='10', mew='2')


ax3 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[5], markers=True,
                  marker="D", dashes=False, label = 'FL t:= 75', linewidth=2,ms='10', mew='2')


ax3 = sns.lineplot(x = loss_prob, y = np.array(error_fl_CO)[6], markers=True,
                  marker="<", dashes=False, label = 'FL t:= 100', linewidth=2,ms='10', mew='2')

# ax.set(ylim=(0.2, 0.5))
# ax.set(xlim=(0.1, 0.9))
ax3.grid(True)
plt.legend(fontsize=11)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE Error",fontsize=14)
plt.title("CO - FL vs TL [Architechture - 1HL/32Neuron]") 
# plt.show(ax3)
plt.savefig(storage_loc+"/CO FL vs TL [Architechture - 4HL-64Neuron] for various t-values.png", format="png")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
plt.figure(figsize=(12, 8))

ax4 = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['PM2.5'])[:], markers=True, 
                  marker="D", dashes=False, label = 'TL Error', linewidth=4, mfc='red', ms='8', mew='2')

ax4 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[0], markers=True,
                  marker="o", dashes=False, label = 'FL t:= 15', linewidth=2,ms='10', mew='2')


ax4 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[1], markers=True,
                  marker="s", dashes=False, label = 'FL t:= 20', linewidth=2,ms='10', mew='2')


ax4 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[2], markers=True,
                  marker="P", dashes=False, label = 'FL t:= 25', linewidth=2,ms='10', mew='2')


ax4 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[3], markers=True,
                  marker="X", dashes=False, label = 'FL t:= 35', linewidth=2,ms='10', mew='2')


ax4 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[4], markers=True,
                  marker="v", dashes=False, label = 'FL t:= 50', linewidth=2,ms='10', mew='2')


ax4 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[5], markers=True,
                  marker="D", dashes=False, label = 'FL t:= 75', linewidth=2,ms='10', mew='2')


ax4 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM25)[6], markers=True,
                  marker="<", dashes=False, label = 'FL t:= 100', linewidth=2,ms='10', mew='2')




# ax.set(ylim=(0.2, 0.5))
# ax.set(ylim=(0.0, 1.0))
ax4.grid(True)
plt.legend(fontsize=11)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE Error",fontsize=14)
plt.title("PM2.5 [Architechture - 4HL/64Neuron]") 
# plt.show(ax4)
plt.savefig(storage_loc+"/PM2.5 FL vs TL [Architechture - 4HL-64Neuron] for various t-values.png", format="png")


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
plt.figure(figsize=(12, 8))

ax5 = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['PM2.5'])[:], markers=True, 
                  marker="D", dashes=False, label = 'TL Error', linewidth=4, mfc='red', ms='8', mew='2')

ax5 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[0], markers=True,
                  marker="o", dashes=False, label = 'FL t:= 15', linewidth=2,ms='10', mew='2')


ax5 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[1], markers=True,
                  marker="s", dashes=False, label = 'FL t:= 20', linewidth=2,ms='10', mew='2')


ax5 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[2], markers=True,
                  marker="P", dashes=False, label = 'FL t:= 25', linewidth=2,ms='10', mew='2')


ax5 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[3], markers=True,
                  marker="X", dashes=False, label = 'FL t:= 35', linewidth=2,ms='10', mew='2')


ax5 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[4], markers=True,
                  marker="v", dashes=False, label = 'FL t:= 50', linewidth=2,ms='10', mew='2')


ax5 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[5], markers=True,
                  marker="D", dashes=False, label = 'FL t:= 75', linewidth=2,ms='10', mew='2')


ax5 = sns.lineplot(x = loss_prob, y = np.array(error_fl_PM10)[6], markers=True,
                  marker="<", dashes=False, label = 'FL t:= 100', linewidth=2,ms='10', mew='2')




# ax.set(ylim=(0.2, 0.5))
# ax.set(ylim=(0.0, 1.0))
ax5.grid(True)
plt.legend(fontsize=11)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE Error",fontsize=14)
plt.title("PM10 [Architechture - 4HL/64Neuron]") 
# plt.show(ax5)
plt.savefig(storage_loc+"/PM10 FL vs TL [Architechture - 4HL-64Neuron] for various t-values.png", format="png")

