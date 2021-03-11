import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM

#Load Deat

#SPECIFY COMPANIES BELOW WHICH ARE INTERESTED
company ='ITC.NS'

#what date or time stamp you want to start your data to analyze
start = dt.datetime(2012,1,1)
end=dt.datetime(2021,2,1)

#load data by saying and using ticketsymbol for the company (you can use google to find ticketsymbol for specific company)
data = web.DataReader(company,'yahoo',start,end)

#Prepare data for Neural Network
#we are going to scale down for all the values we have so that they fit in between 0-1
#if we have lowest price of 10 and highest price of 620 and we are going to press all thoser value between 0 & 1
scaler= MinMaxScaler(feature_range=(0,1))#sklearn preporcessing used here
#we are only going to predict only closing price
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

#how many days we look in past to predict next price
prediction_days = 60

#diffining two empty list
x_train = []
y_train=[]

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x,0])

#now convert those append listing into numpy array
x_train,y_train = np.array(x_train),np.array(y_train)
#reshape x_train so that it worked with neural network
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#build the model
model= Sequential()#basic neural network
#specifying the layers below like one LSTM layer , one dropout layer, then again LSTM layer the again dropout layer to trian the model
#then one dense layer unit to predict stock prices

#note:- more layers the more units we add the longer we going to have the training  
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1 )))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True,))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #for prediction of the next closing value

#to compile the model we are going to use the adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train, epochs=25, batch_size=32)

''' Test the model accuracy on Existing Data '''
#Load Test Data
test_start = dt.datetime(2021,2,1) #this data has not to be seemed by model before
test_end = dt.datetime.now()
#test_end = dt.datetime(2021,3,9)

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#make prediction on test data

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#plot the test predictions
'''plt.plot(actual_prices, color="black",label=f"Actual{company}")
plt.plot(predicted_prices, color= 'green', label=f"Predicted{company}")
plt.title(f"{company} share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} share Price')
plt.legend()
plt.show()'''

#predict next day 
real_data = [model_inputs[len(model_inputs)+1 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"prediction: {prediction}")