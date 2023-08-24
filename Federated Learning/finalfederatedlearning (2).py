# -*- coding: utf-8 -*-




"""### Libraries"""

from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import tensorflow as tf
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



#This will provide commandLine arguments.
import sys
# sys.argv[0] # prints python_script.py
storage_loc =  sys.argv[1]
T = int(sys.argv[2])


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


"""## 1 hidden layer
* with 32 neurons

### traditional Learning
"""

def traditional_ML_1L_32_train(loss_p, gas, train_data):
  # initialize w_0
  model = Sequential()

  # first dense/hidden layer
  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
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

def traditional_ML_1L_32_test(model_name, gas, test_data):

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
  

  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())

  NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))
  return (mse, NE)

# Final plotting
K = 25 # as we have 25 clients
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gases = ['CO', 	'PM10',	'PM2.5']
errors_tl_h1_32 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
models = []
model_name = "TL_model_1L_32_"

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
      model = traditional_ML_1L_32_train(lp, gas, train_data)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = traditional_ML_1L_32_test(model_address, gas, train_data)
      errors_tl_h1_32[gas].append(error)
      # testing model
      # errors_fl_h3_32[gas].append(error_clients_L3_32(model, gas))

"""### Federated Learning"""

# ClientUpdate(k, w): // Run on client k
def Client_L1_32(client_idx, model, gas):
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

# Server executes: 
def Server_L1_32(gas):
    # initialize w_0
    model = Sequential()
    
    # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
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
            n_k[client_idx] = Client_L1_32(client, model, gas) # pass by reference for model
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


    #print(model.get_weights())
    #return error_clients(model, loss_p)) # THIS LINE WON'T BE THERE IN REAL SERVER
    return model

def error_clients_L1_32(model_name, gas, test_data):

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

    mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())
    
    return (mse,NE)

# create an Empty DataFrame object
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
models = []
errors_fl_h1_32 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
gases = ['CO', 	'PM10',	'PM2.5']
model_name = "FL_model_1L_32_"


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

      # Training the model using federated learning
      model = Server_L1_32(gas)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = error_clients_L1_32(model_address, gas, test_data)
      errors_fl_h1_32[gas].append(error)

"""### Plots

#### Original plots
"""
fig, axes = plt.subplots(3,1, figsize=(9,8))

idx = 0

y_arr = [(np.array(errors_tl_h1_32['PM2.5'])[:,1], np.array(errors_fl_h1_32['PM2.5'])[:,1]), (np.array(errors_tl_h1_32['CO'])[:,1], np.array(errors_fl_h1_32['CO'])[:,1]), 
         (np.array(errors_tl_h1_32['PM10'])[:,1], np.array(errors_fl_h1_32['PM10'])[:,1])]
gas = ['PM2.5', 'CO', 'PM10']
# plt.title("With 1 Hidden Layer, the layer with 32 neurons\n\nCO: Loss probabilities vs ERROR Values")

for row in range(3):
    for col in range(1):
        if row == 0:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)
        else:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][0], markers=True, 
                  marker="o", dashes=False, label = 'TL Error', ax = axes[row], linewidth=3, ms='10', mew='2')

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][1], markers=True, 
                  marker="D", dashes=False, label = 'FL Error', ax = axes[row], linewidth=3, ms='10', mew='2')
        
        axes[row].legend(fontsize=12)
        axes[row].grid(True)
        
        if row == 1:
          axes[row].set(ylim=(0.0, 1.0))
          axes[row].set_ylabel('NE', fontsize=14)
        else:
          axes[row].set(ylim=(0.0, 1.0))

        if row == 2:
          pass
          # axes[row].set(xlabel='Loss Probability', fontsize=14)

        
        idx += 1
plt.xlabel("Loss Probability", fontsize=15)
# plt.ylabel("NE Values", loc='center', fontsize=15)
# title = "With 1 Hidden Layer, the layer with 64 neurons"    
# fig.savefig('/content/drive/MyDrive/Colab Notebooks/FL_vs_TL_Plots/{}.png'.format(title))
plt.tight_layout(pad=3);
plt.savefig(storage_loc+"/T={} [Architechture - 1HL-32Neuron].png".format(T), format="png")
plt.savefig(storage_loc+"/T={} [Architechture - 1HL-32Neuron].eps".format(T), format="eps")


"""## 1 hidden layer
* with 64 neurons

### traditional Learning
"""

def traditional_ML_1L_64_train(loss_p, gas, train_data):
  # initialize w_0
  model = Sequential()

  # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
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

def traditional_ML_1L_64_test(model_name, gas, test_data):

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
  

  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())

  NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))
  return (mse, NE)

# Final plotting
K = 25 # as we have 25 clients
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gases = ['CO', 	'PM10',	'PM2.5']
errors_tl_h1_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
models = []
model_name = "TL_model_1L_64_"



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
      model = traditional_ML_1L_64_train(lp, gas, train_data)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = traditional_ML_1L_64_test(model_address, gas, train_data)
      errors_tl_h1_64[gas].append(error)
      # testing model
      # errors_fl_h3_32[gas].append(error_clients_L3_32(model, gas))

"""### Federated Learning"""

# ClientUpdate(k, w): // Run on client k
def Client_L1_64(client_idx, model, gas):
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

# Server executes: 
def Server_L1_64(gas):
    # initialize w_0
    model = Sequential()
    
    # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
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
            n_k[client_idx] = Client_L1_64(client, model, gas) # pass by reference for model
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

def error_clients_L1_64(model_name, gas, test_data):

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

    mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())
    
    return (mse,NE)

# create an Empty DataFrame object
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
models = []
errors_fl_h1_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
gases = ['CO', 	'PM10',	'PM2.5']
model_name = "FL_model_1L_64_"


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

      # Training the model using federated learning
      model = Server_L1_64(gas)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = error_clients_L1_64(model_address, gas, test_data)
      errors_fl_h1_64[gas].append(error)

"""### Plots"""
fig, axes = plt.subplots(3,1, figsize=(9,8))

idx = 0

y_arr = [(np.array(errors_tl_h1_64['PM2.5'])[:,1], np.array(errors_fl_h1_64['PM2.5'])[:,1]), (np.array(errors_tl_h1_64['CO'])[:,1], np.array(errors_fl_h1_64['CO'])[:,1]), 
         (np.array(errors_tl_h1_64['PM10'])[:,1], np.array(errors_fl_h1_64['PM10'])[:,1])]

gas = ['PM2.5', 'CO', 'PM10']

for row in range(3):
    for col in range(1):
        if row == 0:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)
        else:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][0], markers=True, 
                  marker="o", dashes=False, label = 'TL Error', ax = axes[row], linewidth=3, ms='10', mew='2')

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][1], markers=True, 
                  marker="D", dashes=False, label = 'FL Error', ax = axes[row], linewidth=3, ms='10', mew='2')
        
        axes[row].legend(fontsize=12)
        axes[row].grid(True)
        
        if row == 1:
          axes[row].set(ylim=(0.0, 1.0))
          axes[row].set_ylabel('NE', fontsize=14)
        else:
          axes[row].set(ylim=(0.0, 1.0))

        if row == 2:
          pass
          # axes[row].set(xlabel='Loss Probability', fontsize=14)

        
        idx += 1
plt.xlabel("Loss Probability", fontsize=15)
# plt.ylabel("NE Values", loc='center', fontsize=15)
# title = "With 1 Hidden Layer, the layer with 64 neurons"    
# fig.savefig('/content/drive/MyDrive/Colab Notebooks/FL_vs_TL_Plots/{}.png'.format(title))
plt.tight_layout(pad=3);
plt.savefig(storage_loc+"/T={} [Architechture - 1HL-64Neuron].png".format(T), format="png")
plt.savefig(storage_loc+"/T={} [Architechture - 1HL-64Neuron].eps".format(T), format="eps")


"""## 2 hidden layer
* with 32 neurons

### traditional Learning
"""

def traditional_ML_2L_32_train(loss_p, gas, train_data):
  # initialize w_0
  model = Sequential()

  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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

def traditional_ML_2L_32_test(model_name, gas, test_data):

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
  

  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())

  NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))
  return (mse, NE)

# Final plotting
K = 25 # as we have 25 clients
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gases = ['CO', 	'PM10',	'PM2.5']
errors_tl_h2_32 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
models = []
model_name = "TL_model_2L_32_"



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
      model = traditional_ML_2L_32_train(lp, gas, train_data)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = traditional_ML_2L_32_test(model_address, gas, train_data)
      errors_tl_h2_32[gas].append(error)
      # testing model
      # errors_fl_h3_32[gas].append(error_clients_L3_32(model, gas))

"""### Federated Learning"""

# ClientUpdate(k, w): // Run on client k
def Client_L2_32(client_idx, model, gas):
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

# Server executes: 
def Server_L2_32(gas):
    # initialize w_0
    model = Sequential()

    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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
            n_k[client_idx] = Client_L2_32(client, model, gas) # pass by reference for model
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

def error_clients_L2_32(model_name, gas, test_data):

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
    x_test = x_test.reshape(x_test.shape[0],  x_test.shape[1])
    y_test = y_test.reshape(y_test.shape[0],1)

    y_predicted = model.predict(x_test)
    
    from sklearn.metrics import mean_squared_error
    # print("y_predicted", y_predicted)
    # print("y_predicted.ravel()", y_predicted.ravel())
    NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))

    mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())
    
    return (mse,NE)

# create an Empty DataFrame object
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
models = []
errors_fl_h2_32 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
gases = ['CO', 	'PM10',	'PM2.5']
model_name = "FL_model_2L_32_"


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

      # Training the model using federated learning
      model = Server_L2_32(gas)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = error_clients_L2_32(model_address, gas, test_data)
      errors_fl_h2_32[gas].append(error)

"""### Plots"""
#### Customized plots: 0.5"""

fig, axes = plt.subplots(3,1, figsize=(9,8))

idx = 0

y_arr = [(np.array(errors_tl_h2_32['PM2.5'])[:,1], np.array(errors_fl_h2_32['PM2.5'])[:,1]), (np.array(errors_tl_h2_32['CO'])[:,1], np.array(errors_fl_h2_32['CO'])[:,1]), 
         (np.array(errors_tl_h2_32['PM10'])[:,1], np.array(errors_fl_h2_32['PM10'])[:,1])]
gas = ['PM2.5', 'CO', 'PM10']
# plt.title("With 1 Hidden Layer, the layer with 32 neurons\n\nCO: Loss probabilities vs ERROR Values")

for row in range(3):
    for col in range(1):
        if row == 0:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)
        else:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][0], markers=True, 
                  marker="o", dashes=False, label = 'TL Error', ax = axes[row], linewidth=3, ms='10', mew='2')

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][1], markers=True, 
                  marker="D", dashes=False, label = 'FL Error', ax = axes[row], linewidth=3, ms='10', mew='2')
        
        axes[row].legend(fontsize=12)
        axes[row].grid(True)
        
        if row == 1:
          axes[row].set(ylim=(0.0, 1.0))
          axes[row].set_ylabel('NE', fontsize=14)
        else:
          axes[row].set(ylim=(0.0, 1.0))

        if row == 2:
          pass
          # axes[row].set(xlabel='Loss Probability', fontsize=14)

        
        idx += 1
plt.xlabel("Loss Probability", fontsize=15)
# plt.ylabel("NE Values", loc='center', fontsize=15)
# title = "With 1 Hidden Layer, the layer with 64 neurons"    
# fig.savefig('/content/drive/MyDrive/Colab Notebooks/FL_vs_TL_Plots/{}.png'.format(title))
plt.tight_layout(pad=3);
plt.savefig(storage_loc+"/T={} [Architechture - 2HL-32Neuron].png".format(T), format="png")
plt.savefig(storage_loc+"/T={} [Architechture - 2HL-32Neuron].eps".format(T), format="eps")


"""## 2 hidden layers
* with 64 neurons

### traditional Learning
"""

def traditional_ML_2L_64_train(loss_p, gas, train_data):
  # initialize w_0
  # initialize w_0
  model = Sequential()

  # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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

def traditional_ML_2L_64_test(model_name, gas, test_data):

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
  y_test = y_test.reshape(y_test.shape[0], 1)
  
  y_predicted = model.predict(x_test)
  

  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())

  NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))
  return (mse, NE)

# Final plotting
K = 25 # as we have 25 clients
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gases = ['CO', 	'PM10',	'PM2.5']
errors_tl_h2_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
models = []
model_name = "TL_model_2L_64_"



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
      model = traditional_ML_2L_64_train(lp, gas, train_data)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = traditional_ML_2L_64_test(model_address, gas, train_data)
      errors_tl_h2_64[gas].append(error)
      # testing model
      # errors_fl_h3_32[gas].append(error_clients_L3_32(model, gas))

"""### Federated Learning"""

# ClientUpdate(k, w): // Run on client k
def Client_L2_64(client_idx, model, gas):
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

# Server executes: 
def Server_L2_64(gas):
    # initialize w_0
    # initialize w_0
    model = Sequential()

    # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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
            n_k[client_idx] = Client_L2_64(client, model, gas) # pass by reference for model
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

def error_clients_L2_64(model_name, gas, test_data):

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

    mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())
    
    return (mse,NE)

# create an Empty DataFrame object
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
models = []
errors_fl_h2_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
gases = ['CO', 	'PM10',	'PM2.5']
model_name = "FL_model_2L_64_"


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

      # Training the model using federated learning
      model = Server_L2_64(gas)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = error_clients_L2_64(model_address, gas, test_data)
      errors_fl_h2_64[gas].append(error)

"""### Plots"""
fig, axes = plt.subplots(3,1, figsize=(9,8))

idx = 0

y_arr = [(np.array(errors_tl_h2_64['PM2.5'])[:,1], np.array(errors_fl_h2_64['PM2.5'])[:,1]), (np.array(errors_tl_h2_64['CO'])[:,1], np.array(errors_fl_h2_64['CO'])[:,1]), 
         (np.array(errors_tl_h2_64['PM10'])[:,1], np.array(errors_fl_h2_64['PM10'])[:,1])]
gas = ['PM2.5', 'CO', 'PM10']
# plt.title("With 1 Hidden Layer, the layer with 32 neurons\n\nCO: Loss probabilities vs ERROR Values")

for row in range(3):
    for col in range(1):
        if row == 0:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)
        else:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][0], markers=True, 
                  marker="o", dashes=False, label = 'TL Error', ax = axes[row], linewidth=3, ms='10', mew='2')

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][1], markers=True, 
                  marker="D", dashes=False, label = 'FL Error', ax = axes[row], linewidth=3, ms='10', mew='2')
        
        axes[row].legend(fontsize=12)
        axes[row].grid(True)
        
        if row == 1:
          axes[row].set(ylim=(0.0, 1.0))
          axes[row].set_ylabel('NE', fontsize=14)
        else:
          axes[row].set(ylim=(0.0, 1.0))

        if row == 2:
          pass
          # axes[row].set(xlabel='Loss Probability', fontsize=14)

        
        idx += 1
plt.xlabel("Loss Probability", fontsize=15)
# plt.ylabel("NE Values", loc='center', fontsize=15)
# title = "With 1 Hidden Layer, the layer with 64 neurons"    
# fig.savefig('/content/drive/MyDrive/Colab Notebooks/FL_vs_TL_Plots/{}.png'.format(title))
plt.tight_layout(pad=3);
plt.savefig(storage_loc+"/T={} [Architechture - 2HL-64Neuron].png".format(T), format="png")
plt.savefig(storage_loc+"/T={} [Architechture - 2HL-64Neuron].eps".format(T), format="eps")


"""## 3 hidden layer
* with 32 neurons

### traditional Learning
"""

def traditional_ML_3L_32_train(loss_p, gas, train_data):
  # initialize w_0
  model = Sequential()

  # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
  model.add(Dropout(0.1))

  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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

def traditional_ML_3L_32_test(model_name, gas, test_data):

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
  

  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())

  NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))
  return (mse, NE)

# Final plotting
K = 25 # as we have 25 clients
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gases = ['CO', 	'PM10',	'PM2.5']
errors_tl_h3_32 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
models = []
model_name = "TL_model_3L_32_"



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
      model = traditional_ML_3L_32_train(lp, gas, train_data)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = traditional_ML_3L_32_test(model_address, gas, train_data)
      errors_tl_h3_32[gas].append(error)
      # testing model
      # errors_fl_h3_32[gas].append(error_clients_L3_32(model, gas))

"""### Federated Learning"""

# ClientUpdate(k, w): // Run on client k
def Client_L3_32(client_idx, model, gas):
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

# Server executes: 
def Server_L3_32(gas):
    # initialize w_0
    model = Sequential()

    # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
    model.add(Dropout(0.1))

    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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
            n_k[client_idx] = Client_L3_32(client, model, gas) # pass by reference for model
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

def error_clients_L3_32(model_name, gas, test_data):

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

    mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())
    
    return (mse,NE)

# create an Empty DataFrame object
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
models = []
errors_fl_h3_32 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
gases = ['CO', 	'PM10',	'PM2.5']
model_name = "FL_model_3L_32_"


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

      # # Training the model using federated learning
      model = Server_L3_32(gas)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = error_clients_L3_32(model_address, gas, test_data)
      errors_fl_h3_32[gas].append(error)

"""### Plots"""
#### Customized plots: 0.5"""
fig, axes = plt.subplots(3,1, figsize=(9,8))

idx = 0

y_arr = [(np.array(errors_tl_h3_32['PM2.5'])[:,1], np.array(errors_fl_h3_32['PM2.5'])[:,1]), (np.array(errors_tl_h3_32['CO'])[:,1], np.array(errors_fl_h3_32['CO'])[:,1]), 
         (np.array(errors_tl_h3_32['PM10'])[:,1], np.array(errors_fl_h3_32['PM10'])[:,1])]
gas = ['PM2.5', 'CO', 'PM10']
# plt.title("With 1 Hidden Layer, the layer with 32 neurons\n\nCO: Loss probabilities vs ERROR Values")

for row in range(3):
    for col in range(1):
        if row == 0:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)
        else:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][0], markers=True, 
                  marker="o", dashes=False, label = 'TL Error', ax = axes[row], linewidth=3, ms='10', mew='2')

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][1], markers=True, 
                  marker="D", dashes=False, label = 'FL Error', ax = axes[row], linewidth=3, ms='10', mew='2')
        
        axes[row].legend(fontsize=12)
        axes[row].grid(True)
        
        if row == 1:
          axes[row].set(ylim=(0.0, 1.0))
          axes[row].set_ylabel('NE', fontsize=14)
        else:
          axes[row].set(ylim=(0.0, 1.0))

        if row == 2:
          pass
          # axes[row].set(xlabel='Loss Probability', fontsize=14)

        
        idx += 1
plt.xlabel("Loss Probability", fontsize=15)
# plt.ylabel("NE Values", loc='center', fontsize=15)
# title = "With 1 Hidden Layer, the layer with 64 neurons"    
# fig.savefig('/content/drive/MyDrive/Colab Notebooks/FL_vs_TL_Plots/{}.png'.format(title))
plt.tight_layout(pad=3);
plt.savefig(storage_loc+"/T={} [Architechture - 3HL-32Neuron].png".format(T), format="png")
plt.savefig(storage_loc+"/T={} [Architechture - 3HL-32Neuron].eps".format(T), format="eps")


"""## 3 hidden layers
* with 64 neurons

### traditional Learning
"""

def traditional_ML_3L_64_train(loss_p, gas, train_data):
  # initialize w_0
  # initialize w_0
  model = Sequential()

  # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
  model.add(Dropout(0.1))

  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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

def traditional_ML_3L_64_test(model_name, gas, test_data):

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
  x_test = x_test.reshape(x_test.shape[0],  x_test.shape[1])
  y_test = y_test.reshape(y_test.shape[0],1)
  
  y_predicted = model.predict(x_test)
  

  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())

  NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))
  return (mse, NE)

# Final plotting
K = 25 # as we have 25 clients
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gases = ['CO', 	'PM10',	'PM2.5']
errors_tl_h3_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
models = []
model_name = "TL_model_3L_64_"



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
      model = traditional_ML_3L_64_train(lp, gas, train_data)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = traditional_ML_3L_64_test(model_address, gas, train_data)
      errors_tl_h3_64[gas].append(error)
      # testing model
      # errors_fl_h3_32[gas].append(error_clients_L3_32(model, gas))

"""### Federated Learning"""

# ClientUpdate(k, w): // Run on client k
def Client_L3_64(client_idx, model, gas):
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

# Server executes: 
def Server_L3_64(gas):
    # initialize w_0
    # initialize w_0
    model = Sequential()


    # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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
            n_k[client_idx] = Client_L3_64(client, model, gas) # pass by reference for model
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

def error_clients_L3_64(model_name, gas, test_data):

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
    x_test = x_test.reshape(x_test.shape[0],  x_test.shape[1])
    y_test = y_test.reshape(y_test.shape[0],1)

    y_predicted = model.predict(x_test)
    
    from sklearn.metrics import mean_squared_error
    # print("y_predicted", y_predicted)
    # print("y_predicted.ravel()", y_predicted.ravel())
    NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))

    mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())
    
    return (mse,NE)

# create an Empty DataFrame object
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
models = []
errors_fl_h3_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
gases = ['CO', 	'PM10',	'PM2.5']
model_name = "FL_model_3L_64_"


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

      # Training the model using federated learning
      model = Server_L3_64(gas)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = error_clients_L3_64(model_address, gas, test_data)
      errors_fl_h3_64[gas].append(error)

"""### Plots"""


fig, axes = plt.subplots(3,1, figsize=(9,8))

idx = 0

y_arr = [(np.array(errors_tl_h3_64['PM2.5'])[:,1], np.array(errors_fl_h3_64['PM2.5'])[:,1]), (np.array(errors_tl_h3_64['CO'])[:,1], np.array(errors_fl_h3_64['CO'])[:,1]), 
         (np.array(errors_tl_h3_64['PM10'])[:,1], np.array(errors_fl_h3_64['PM10'])[:,1])]
gas = ['PM2.5', 'CO', 'PM10']
# plt.title("With 1 Hidden Layer, the layer with 32 neurons\n\nCO: Loss probabilities vs ERROR Values")

# plt.title("With 1 Hidden Layer, the layer with 32 neurons\n\nCO: Loss probabilities vs ERROR Values")

for row in range(3):
    for col in range(1):
        if row == 0:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)
        else:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][0], markers=True, 
                  marker="o", dashes=False, label = 'TL Error', ax = axes[row], linewidth=3, ms='10', mew='2')

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][1], markers=True, 
                  marker="D", dashes=False, label = 'FL Error', ax = axes[row], linewidth=3, ms='10', mew='2')
        
        axes[row].legend(fontsize=12)
        axes[row].grid(True)
        
        if row == 1:
          axes[row].set(ylim=(0.0, 1.0))
          axes[row].set_ylabel('NE', fontsize=14)
        else:
          axes[row].set(ylim=(0.0, 1.0))

        if row == 2:
          pass
          # axes[row].set(xlabel='Loss Probability', fontsize=14)

        
        idx += 1
plt.xlabel("Loss Probability", fontsize=15)
# plt.ylabel("NE Values", loc='center', fontsize=15)
# title = "With 1 Hidden Layer, the layer with 64 neurons"    
# fig.savefig('/content/drive/MyDrive/Colab Notebooks/FL_vs_TL_Plots/{}.png'.format(title))
plt.tight_layout(pad=3);
plt.savefig(storage_loc+"/T={} [Architechture - 3HL-64Neuron].png".format(T), format="png")
plt.savefig(storage_loc+"/T={} [Architechture - 3HL-64Neuron].eps".format(T), format="eps")



"""## 4 hidden layer
* with 32 neurons

### traditional Learning
"""

def traditional_ML_4L_32_train(loss_p, gas, train_data):
  # initialize w_0
  model = Sequential()

  # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
  model.add(Dropout(0.1))

  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
  model.add(Dropout(0.1))

  model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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

def traditional_ML_4L_32_test(model_name, gas, test_data):

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
  x_test = x_test.reshape(x_test.shape[0],  x_test.shape[1])
  y_test = y_test.reshape(y_test.shape[0],1)
  
  y_predicted = model.predict(x_test)
  

  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())

  NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))
  return (mse, NE)

# Final plotting
K = 25 # as we have 25 clients
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gases = ['CO', 	'PM10',	'PM2.5']
errors_tl_h4_32 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
models = []
model_name = "TL_model_4L_32_"

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
      model = traditional_ML_4L_32_train(lp, gas, train_data)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = traditional_ML_4L_32_test(model_address, gas, train_data)
      errors_tl_h4_32[gas].append(error)
      # testing model
      # errors_fl_h3_32[gas].append(error_clients_L3_32(model, gas))

"""### Federated Learning"""

# ClientUpdate(k, w): // Run on client k
def Client_L4_32(client_idx, model, gas):
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

# Server executes: 
def Server_L4_32(gas):
    # initialize w_0
    model = Sequential()

    # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
    model.add(Dropout(0.1))

    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
    model.add(Dropout(0.1))

    model.add(Dense(32, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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
            n_k[client_idx] = Client_L4_32(client, model, gas) # pass by reference for model
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

def error_clients_L4_32(model_name, gas, test_data):

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
    x_test = x_test.reshape(x_test.shape[0],  x_test.shape[1])
    y_test = y_test.reshape(y_test.shape[0],1)

    y_predicted = model.predict(x_test)
    
    from sklearn.metrics import mean_squared_error
    # print("y_predicted", y_predicted)
    # print("y_predicted.ravel()", y_predicted.ravel())
    NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))

    mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())
    
    return (mse,NE)

# create an Empty DataFrame object
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
models = []
errors_fl_h4_32 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
gases = ['CO', 	'PM10',	'PM2.5']
model_name = "FL_model_4L_32_"


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

      # # Training the model using federated learning
      model = Server_L4_32(gas)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = error_clients_L4_32(model_address, gas, test_data)
      errors_fl_h4_32[gas].append(error)

"""### Plots"""

fig, axes = plt.subplots(3,1, figsize=(9,8))

idx = 0

y_arr = [(np.array(errors_tl_h4_32['PM2.5'])[:,1], np.array(errors_fl_h4_32['PM2.5'])[:,1]), (np.array(errors_tl_h4_32['CO'])[:,1], np.array(errors_fl_h4_32['CO'])[:,1]), 
         (np.array(errors_tl_h4_32['PM10'])[:,1], np.array(errors_fl_h4_32['PM10'])[:,1])]
gas = ['PM2.5', 'CO', 'PM10']
# plt.title("With 1 Hidden Layer, the layer with 32 neurons\n\nCO: Loss probabilities vs ERROR Values")

for row in range(3):
    for col in range(1):
        if row == 0:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)
        else:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][0], markers=True, 
                  marker="o", dashes=False, label = 'TL Error', ax = axes[row], linewidth=3, ms='10', mew='2')

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][1], markers=True, 
                  marker="D", dashes=False, label = 'FL Error', ax = axes[row], linewidth=3, ms='10', mew='2')
        
        axes[row].legend(fontsize=12)
        axes[row].grid(True)
        
        if row == 1:
          axes[row].set(ylim=(0.0, 1.0))
          axes[row].set_ylabel('NE', fontsize=14)
        else:
          axes[row].set(ylim=(0.0, 1.0))

        if row == 2:
          pass
          # axes[row].set(xlabel='Loss Probability', fontsize=14)

        
        idx += 1
plt.xlabel("Loss Probability", fontsize=15)
# # plt.ylabel("NE Values", loc='center', fontsize=15)
# title = "With 1 Hidden Layer, the layer with 64 neurons"    
# fig.savefig('/content/drive/MyDrive/Colab Notebooks/FL_vs_TL_Plots/{}.png'.format(title))
plt.tight_layout(pad=3);
plt.savefig(storage_loc+"/T={} [Architechture - 4HL-32Neuron].png".format(T), format="png")
plt.savefig(storage_loc+"/T={} [Architechture - 4HL-32Neuron].eps".format(T), format="eps")


"""## 4 hidden layers
* with 64 neurons

### traditional Learning
"""

def traditional_ML_4L_64_train(loss_p, gas, train_data):
  # initialize w_0
  # initialize w_0
  model = Sequential()

  # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
  model.add(Dropout(0.1))

  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
  model.add(Dropout(0.1))

  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
  model.add(Dropout(0.1))

  model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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
  x_test = x_test.reshape(x_test.shape[0],  x_test.shape[1])
  y_test = y_test.reshape(y_test.shape[0],1)
  
  y_predicted = model.predict(x_test)
  

  from sklearn.metrics import mean_squared_error
  mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())

  NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))
  return (mse, NE)

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
      model = traditional_ML_4L_64_train(lp, gas, train_data)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = traditional_ML_4L_64_test(model_address, gas, train_data)
      errors_tl_h4_64[gas].append(error)
      # testing model
      # errors_fl_h3_32[gas].append(error_clients_L3_32(model, gas))

"""### Federated Learning"""

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

# Server executes: 
def Server_L4_64(gas):
    # initialize w_0
    # initialize w_0
    model = Sequential()


    # model.add(Dense(64, kernel_initializer='zeros', bias_initializer='zeros', activation=tf.nn.relu, input_dim=6)) # first dense/hidden layer
    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu, input_dim=8))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
    model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='he_uniform', bias_initializer="zeros", activation=tf.nn.relu))
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
    x_test = x_test.reshape(x_test.shape[0],   x_test.shape[1])
    y_test = y_test.reshape(y_test.shape[0],1)

    y_predicted = model.predict(x_test)
    
    from sklearn.metrics import mean_squared_error
    # print("y_predicted", y_predicted)
    # print("y_predicted.ravel()", y_predicted.ravel())
    NE = np.sum(abs(np.array(y_test) - np.array(y_predicted))) / np.sum(np.array(y_test))

    mse = mean_squared_error(y_predicted.ravel(), y_test.ravel())
    
    return (mse,NE)

# create an Empty DataFrame object
train_data = pd.DataFrame()
test_data = pd.DataFrame()
loss_prob = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
models = []
errors_fl_h4_64 = {'CO':[], 	'PM10':[],	'PM2.5':[]}
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

      # Training the model using federated learning
      model = Server_L4_64(gas)

      ## Saving model
      ## serializing model to JSON
      model_address = storage_loc + "/" + model_name + gas + str(lp) + "_"


      model_json = model.to_json()
      with open(model_address + ".json", "w") as json_file:
        json_file.write(model_json)

      # serialize weights to HDF5
      model.save_weights(model_address + ".h5")


      ## Testing model
      error = error_clients_L4_64(model_address, gas, test_data)
      errors_fl_h4_64[gas].append(error)

"""### Plots"""



fig, axes = plt.subplots(3,1, figsize=(9,8))

idx = 0
y_arr = [(np.array(errors_tl_h4_64['PM2.5'])[:,1], np.array(errors_fl_h4_64['PM2.5'])[:,1]), (np.array(errors_tl_h4_64['CO'])[:,1], np.array(errors_fl_h4_64['CO'])[:,1]), 
         (np.array(errors_tl_h4_64['PM10'])[:,1], np.array(errors_fl_h4_64['PM10'])[:,1])]
gas = ['PM2.5', 'CO', 'PM10']
# plt.title("With 1 Hidden Layer, the layer with 32 neurons\n\nCO: Loss probabilities vs ERROR Values")

for row in range(3):
    for col in range(1):
        if row == 0:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)
        else:
          axes[row].set_title("{}".format(gas[row]), fontsize=13)

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][0], markers=True, 
                  marker="o", dashes=False, label = 'TL Error', ax = axes[row], linewidth=3, ms='10', mew='2')

        axes[row] = sns.lineplot(x = loss_prob, y = y_arr[row][1], markers=True, 
                  marker="D", dashes=False, label = 'FL Error', ax = axes[row], linewidth=3, ms='10', mew='2')
        
        axes[row].legend(fontsize=12)
        axes[row].grid(True)
        
        if row == 1:
          axes[row].set(ylim=(0.0, 1.0))
          axes[row].set_ylabel('NE', fontsize=14)
        else:
          axes[row].set(ylim=(0.0, 1.0))

        if row == 2:
          pass
          # axes[row].set(xlabel='Loss Probability', fontsize=14)

        
        idx += 1
plt.xlabel("Loss Probability", fontsize=15)
# # plt.ylabel("NE Values", loc='center', fontsize=15)
# title = "With 1 Hidden Layer, the layer with 64 neurons"    
# fig.savefig('/content/drive/MyDrive/Colab Notebooks/FL_vs_TL_Plots/{}.png'.format(title))
plt.tight_layout(pad=3);
plt.savefig(storage_loc+"/T={} [Architechture - 4HL-64Neuron].png".format(T), format="png")
plt.savefig(storage_loc+"/T={} [Architechture - 4HL-64Neuron].eps".format(T), format="eps")


"""## Improvement in Learning through architecture

### with number of hidden layers
"""

import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
# plt.figure(figsize=(6,6))
ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_32['CO'])[:,1], markers=True, 
                  marker="o", dashes=False, label = 'TL: 1 HL', linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_32['CO'])[:,1], markers=True, 
                  marker="s", dashes=False, label = 'TL: 2 HL', linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_32['CO'])[:,1], markers=True, 
                  marker="P", dashes=False, label = 'TL: 3 HL', linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_32['CO'])[:,1], markers=True, 
                  marker="X", dashes=False, label = 'TL: 4 HL', linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.2, 0.5))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Traditional learning errors using varying layers with 32 neurons\nCO: Loss probabilities vs ERROR Values") 
plt.title("Traditional learning - CO")
#plt.show(ax)

# traditional learning perfromed best with 4 hidden layer


import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_32['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_32['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_32['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 3 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_32['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 4 HL',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.0, 1.0))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Traditional learning errors using varying layers with 32 neurons\nPM2.5: Loss probabilities vs ERROR Values")
plt.title("Traditional learning - PM2.5")
#plt.show(ax)


import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_32['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_32['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_32['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 3 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_32['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 4 HL',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.0, 1.0))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Traditional learning errors using varying layers with 32 neurons\nPM10: Loss probabilities vs ERROR Values")
plt.title("Traditional learning - PM10")
#plt.show(ax)

import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_32['CO'])[:,1], markers=True, 
                  marker="D", dashes=False, label = 'FL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_32['CO'])[:,1], markers=True, 
                  marker="D", dashes=False, label = 'FL: 2 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_32['CO'])[:,1], markers=True, 
                  marker="D", dashes=False, label = 'FL: 3 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_32['CO'])[:,1], markers=True, 
                  marker="D", dashes=False, label = 'FL: 4 HL',linewidth=1, ms='10', mew='2')
ax.set(ylim=(0.2, 0.5))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Federated learning errors using varying layers with 32 neurons\nCO: Loss probabilities vs ERROR Values") 
plt.title("Federated learning - CO")
#plt.show(ax)

## FL performed best with 3 hidden layer 64 neurons

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_32['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 1 HL')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_32['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_32['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 3 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_32['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 4 HL',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.0, 1.0))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Federated learning errors using varying layers with 32 neurons\nPM2.5: Loss probabilities vs ERROR Values")
plt.title("Federated learning - PM2.5")
#plt.show(ax)


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_32['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 1 HL')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_32['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_32['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 3 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_32['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 4 HL',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.0, 1.0))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Federated learning errors using varying layers with 32 neurons\nPM10: Loss probabilities vs ERROR Values")
plt.title("Federated learning - PM10")
#plt.show(ax)

import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_64['CO'])[:,1], markers=True, 
                  marker="o", dashes=False, label = 'TL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_64['CO'])[:,1], markers=True, 
                  marker="o", dashes=False, label = 'TL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_64['CO'])[:,1], markers=True, 
                  marker="o", dashes=False, label = 'TL: 3 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['CO'])[:,1], markers=True, 
                  marker="o", dashes=False, label = 'TL: 4 HL',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.2, 0.5))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Traditional learning errors using varying layers with 64 neurons\nCO: Loss probabilities vs ERROR Values") 
plt.title("Traditional learning - CO")
#plt.show(ax)

# traditional learning perfromed best with 4 hidden layer

import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_64['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_64['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_64['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 3 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 4 HL',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.3, 0.7))
ax.grid(True)
plt.legend(fontsize=12)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Traditional learning errors using varying layers with 64 neurons\nPM2.5: Loss probabilities vs ERROR Values")
plt.title("PM2.5",fontsize=14)
#plt.show(ax)


import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_64['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_64['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_64['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 3 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 4 HL',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.3, 0.8))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Traditional learning errors using varying layers with 64 neurons\nPM10: Loss probabilities vs ERROR Values")
plt.title("Traditional learning - PM10")
#plt.show(ax)

import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_64['CO'])[:,1], markers=True, 
                  marker="D", dashes=False, label = 'FL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_64['CO'])[:,1], markers=True, 
                  marker="D", dashes=False, label = 'FL: 2 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_64['CO'])[:,1], markers=True, 
                  marker="D", dashes=False, label = 'FL: 3 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_64['CO'])[:,1], markers=True, 
                  marker="D", dashes=False, label = 'FL: 4 HL',linewidth=1, ms='10', mew='2')
ax.set(ylim=(0.2, 0.5))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Federated learning errors using varying layers with 64 neurons\nCO: Loss probabilities vs ERROR Values") 
plt.title("Federated learning - CO")
#plt.show(ax)

## FL performed best with 3 hidden layer 64 neurons

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_64['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_64['PM2.5'])[:,1], markers=True,
                  marker="s", dashes=False, label = 'FL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_64['PM2.5'])[:,1], markers=True,
                  marker="P", dashes=False, label = 'FL: 3 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_64['PM2.5'])[:,1], markers=True,
                  marker="X", dashes=False, label = 'FL: 4 HL',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.4, 0.8))
ax.grid(True)
plt.legend(fontsize=12)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Federated learning errors using varying layers with 64 neurons\n'PM2.5': Loss probabilities vs ERROR Values")
plt.title("PM2.5", fontsize=14)
#plt.show(ax)


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_64['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_64['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_64['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 3 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_64['PM10'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 4 HL',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.3, 0.8))
ax.grid(True)
plt.legend(fontsize=11)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Federated learning errors using varying layers with 64 neurons \n'PM10': Loss probabilities vs ERROR Values")
plt.title("Federated learning - PM10")
#plt.show(ax)

"""### chosen for paper"""

import seaborn as sns
import matplotlib.pyplot as plt

# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_64['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'TL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_64['PM2.5'])[:,1], markers=True,
                  marker="s", dashes=False, label = 'TL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_64['PM2.5'])[:,1], markers=True,
                  marker="P", dashes=False, label = 'TL: 3 HL',linewidth=1, ms='12', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['PM2.5'])[:,1], markers=True,
                  marker="X", dashes=False, label = 'TL: 4 HL',linewidth=1, ms='12', mew='2')

ax.set(ylim=(0.0, 1.0))
ax.grid(True)
plt.legend(fontsize=12)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Traditional learning errors using varying layers with 64 neurons\nPM2.5: Loss probabilities vs ERROR Values")
plt.title("TL - PM2.5",fontsize=14)
#plt.show(ax)
plt.savefig(storage_loc+"/T={} TL - PM2.5 TL various layers.png".format(T), format="png")
plt.savefig(storage_loc+"/T={} TL - PM2.5 TL various layers.eps".format(T), format="eps")




import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_64['PM2.5'])[:,1], markers=True,
                  marker="o", dashes=False, label = 'FL: 1 HL',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_64['PM2.5'])[:,1], markers=True,
                  marker="s", dashes=False, label = 'FL: 2 HL',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_64['PM2.5'])[:,1], markers=True,
                  marker="P", dashes=False, label = 'FL: 3 HL',linewidth=1, ms='12', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_64['PM2.5'])[:,1], markers=True,
                  marker="X", dashes=False, label = 'FL: 4 HL',linewidth=1, ms='12', mew='2')

ax.set(ylim=(0.0, 1.0))
ax.grid(True)
plt.legend(fontsize=12)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
# plt.title("Federated learning errors using varying layers with 64 neurons\n'PM2.5': Loss probabilities vs ERROR Values")
plt.title("FL - PM2.5", fontsize=14)
plt.savefig(storage_loc+"/T={} FL - PM2.5 various layers.png".format(T), format="png")
plt.savefig(storage_loc+"/T={} FL - PM2.5 various layers.eps".format(T), format="eps")
#plt.show(ax)



"""### with number of neurons in hidden layers"""

import seaborn as sns
import matplotlib.pyplot as plt

# plt.figure(figsize=(12,8))
# sns.lineplot(x = loss_prob, y = errors_tl, markers=True, dashes=False)
ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_32['CO'])[:,1], markers=True, 
                  marker="o", color='b', dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_64['CO'])[:,1], markers=True, 
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_32['CO'])[:,1], markers=True, 
                  marker="o", color='g', mfc='red', dashes=False, label = 'FL: 32 Neuron', linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_64['CO'])[:,1], markers=True, 
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.0, 1.0))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 1 HL\nCO: Loss probabilities vs ERROR Values") 
#plt.show(ax)

# traditional learning perfromed best with 4 hidden layer
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_32['PM2.5'])[:,1], markers=True,
                  marker="o", color='b', dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_64['PM2.5'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_32['PM2.5'])[:,1], markers=True,
                  marker="o", color='g', mfc='red', dashes=False, label = 'FL: 32 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_64['PM2.5'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.4, 0.8))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 1 HL\n'PM2.5': Loss probabilities vs ERROR Values")
#plt.show(ax)


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_32['PM10'])[:,1], markers=True,
                  marker="o", color='b', mfc='red',  dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h1_64['PM10'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_32['PM10'])[:,1], markers=True,
                  marker="o", color='g', mfc='red',  dashes=False, label = 'FL: 32 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h1_64['PM10'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.4, 0.8))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 1 HL\n'PM10': Loss probabilities vs ERROR Values")
#plt.show(ax)

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_32['CO'])[:,1], markers=True,
                  marker="o", color='b', dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_64['CO'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_32['CO'])[:,1], markers=True,
                  marker="o", color='g', mfc='red',  dashes=False, label = 'FL: 32 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_64['CO'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.2, 0.5))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 2 HL\n'CO': Loss probabilities vs ERROR Values")
#plt.show(ax)




import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_32['PM2.5'])[:,1], markers=True,
                  marker="o", color='b', mfc='red',  dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_64['PM2.5'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_32['PM2.5'])[:,1], markers=True,
                  marker="o", color='g', mfc='red', dashes=False, label = 'FL: 32 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_64['PM2.5'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.4, 0.7))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 2 HL\n'PM2.5': Loss probabilities vs ERROR Values")
#plt.show(ax)



import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_32['PM10'])[:,1], markers=True,
                  marker="o", color='b', mfc='red', dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h2_64['PM10'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_32['PM10'])[:,1], markers=True,
                  marker="o", color='g', mfc='red',  dashes=False, label = 'FL: 32 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h2_64['PM10'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.4, 0.7))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 2 HL\n'PM10': Loss probabilities vs ERROR Values")
#plt.show(ax)

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_32['CO'])[:,1], markers=True,
                  marker="o", color='b', mfc='red',  dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_64['CO'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_32['CO'])[:,1], markers=True,
                  marker="o", color='g', mfc='red',  dashes=False, label = 'FL: 32 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_64['CO'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.1, 0.4))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 3 HL\n'CO': Loss probabilities vs ERROR Values")
#plt.show(ax)


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_32['PM2.5'])[:,1], markers=True,
                  marker="o", color='b', mfc='red', dashes=False, label = 'TL: 32 Neuron',linewidth=2, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_64['PM2.5'])[:,1], markers=True,
                  marker="s", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=2, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_32['PM2.5'])[:,1], markers=True,
                  marker="P", color='g', mfc='red',  dashes=False, label = 'FL: 32 Neuron',linewidth=2, ms='12', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_64['PM2.5'])[:,1], markers=True,
                  marker="X", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=2, ms='12', mew='2')

ax.set(ylim=(0.35, 0.65))
ax.grid(True)
plt.legend(fontsize=12)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("PM2.5", fontsize=13)
plt.title("FL - PM2.5", fontsize=14)
plt.savefig(storage_loc+"/T={} FL vs TL - PM2.5 same number of layers but different neurons.png".format(T), format="png")
plt.savefig(storage_loc+"/T={} FL vs TL - PM2.5 same number of layers but different neurons.eps".format(T), format="eps")
#plt.show(ax)


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_32['PM10'])[:,1], markers=True,
                  marker="o", color='b', mfc='red',  dashes=False, label = 'TL: 32 Neuron',linewidth=2, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h3_64['PM10'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=2, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_32['PM10'])[:,1], markers=True,
                  marker="o", color='g', mfc='red', dashes=False, label = 'FL: 32 Neuron',linewidth=2, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h3_64['PM10'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=2, ms='10', mew='2')

ax.set(ylim=(0.3, 0.7))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 3 HL\n'PM10': Loss probabilities vs ERROR Values")
#plt.show(ax)

import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_32['CO'])[:,1], markers=True,
                  marker="o", color='b', mfc='red', dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['CO'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_32['CO'])[:,1], markers=True,
                  marker="o", color='g', mfc='red',  dashes=False, label = 'FL: 32 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_64['CO'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.1, 0.5))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 4 HL\n'CO': Loss probabilities vs ERROR Values")
#plt.show(ax)


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_32['PM2.5'])[:,1], markers=True,
                  marker="o", color='b', mfc='red',  dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['PM2.5'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_32['PM2.5'])[:,1], markers=True,
                  marker="o", color='g', mfc='red',  dashes=False, label = 'FL: 32 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_64['PM2.5'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.3, 0.7))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 4 HL\n'PM2.5': Loss probabilities vs ERROR Values")
#plt.show(ax)


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_32['PM10'])[:,1], markers=True,
                  marker="o", color='b', mfc='red',  dashes=False, label = 'TL: 32 Neuron',linewidth=1, ms='10', mew='2')

ax = sns.lineplot(x = loss_prob, y = np.array(errors_tl_h4_64['PM10'])[:,1], markers=True,
                  marker="D", color='b', dashes=False, label = 'TL: 64 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_32['PM10'])[:,1], markers=True,
                  marker="o", color='g', mfc='red', dashes=False, label = 'FL: 32 Neuron',linewidth=1, ms='10', mew='2')


ax = sns.lineplot(x = loss_prob, y = np.array(errors_fl_h4_64['PM10'])[:,1], markers=True,
                  marker="D", color='g', dashes=False, label = 'FL: 64 Neuron',linewidth=1, ms='10', mew='2')

ax.set(ylim=(0.3, 0.7))
ax.grid(True)
plt.legend(fontsize=14)
plt.xlabel("Loss Probability", fontsize=14)
plt.ylabel("NE",fontsize=14)
plt.title("Traditional vs Federated learning number of neurons in 4 HL\n'PM10': Loss probabilities vs ERROR Values")
# plt.title("PM10:")
#plt.show(ax)

