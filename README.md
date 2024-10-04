# Stock Price Prediction

## AIM
To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Develop a Recurrent Neural Network (RNN) model to predict the stock prices of Google. The goal is to train the model using historical stock price data and then evaluate its performance on a separate test dataset

Dataset: The dataset consists of two CSV files: Trainset.csv: This file contains historical stock price data of Google, which will be used for training the RNN model. It includes features such as the opening price of the stock. Testset.csv: This file contains additional historical stock price data of Google, which will be used for testing the trained RNN model. Similarly, it includes features such as the opening price of the stock.

The objective is to build a model that can effectively learn from the patterns in the training data to make accurate predictions on the test data.

## Design Steps

### Step 1:
Read and preprocess training data, including scaling and sequence creation.

### Step 2:
Initialize a Sequential model and add SimpleRNN and Dense layers.

### Step 3:
Compile the model with Adam optimizer and mean squared error loss.

### Step 4:
Train the model on the prepared training data.

### Step 5:
process test data, predict using the trained model, and visualize the results

## Program
#### Name:ANUSSHARON.S
#### Register Number:212222240010
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('/content/trainset (1).csv')

dataset_train.columns

dataset_train.head()
train_set = dataset_train.iloc[:,1:2].valuestype(train_set)
type(train_set)
train_set.shape
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_train.shape
length = 60
n_features = 1
model = Sequential()
model.compile(optimizer='adam', loss='mean_squared_error')
print("Name: ANUSHARON.S      Register Number: 212222240010        ")
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
dataset_test = pd.read_csv('/content/testset (1).csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
X_test.shape
print("Name: ANUSHARON.S     Register Number: 212222240010    ")
plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```







## Output

### True Stock Price, Predicted Stock Price vs time
![Screenshot 2024-10-04 104604](https://github.com/user-attachments/assets/408a1838-9800-44e8-9f1d-3c785c8aa894)

![Screenshot 2024-10-04 104801](https://github.com/user-attachments/assets/9b8101e6-5c38-4493-a86d-835929cabe57)


### Mean Square Error

![Screenshot 2024-10-04 105219](https://github.com/user-attachments/assets/e5e2053f-b0fb-4889-b3a8-4d98523dd0f6)

## Result
Thus a Recurrent Neural Network model for stock price prediction is done.


