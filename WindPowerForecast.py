# Import standard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, DateFormatter

# Import keras
from keras.models import Sequential
from keras.layers import InputLayer, SimpleRNN, Dense, Dropout
from keras.optimizers import RMSprop

# Import Scikit-Learn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Import data
filename1 = 'TrainData.csv'
filename2 = 'Solution.csv'
filename3 = 'WeatherForecastInput.csv'

data = pd.read_csv(filename1, parse_dates=[0])
solution = pd.read_csv(filename2, parse_dates=[0])
weather_forecast = pd.read_csv(filename3, parse_dates=[0])


# setting index
data.set_index(['TIMESTAMP'], inplace=True)
solution.set_index(['TIMESTAMP'], inplace=True)
weather_forecast.set_index(['TIMESTAMP'], inplace=True)

# Setting understandable feature names
data['windspeed'] = data['WS10']
data['zonal'] = data['U10']
data['meridional'] = data['V10']
data.drop(columns=['U10','V10','WS10','U100','V100','WS100'], inplace=True)

weather_forecast['windspeed'] = weather_forecast['WS10']
weather_forecast['zonal'] = weather_forecast['U10']
weather_forecast['meridional'] = weather_forecast['V10']

drop = ['U10','V10','WS10','U100','V100','WS100']
for i in drop: 
    if i in weather_forecast.columns:
        weather_forecast.drop(columns=i, inplace=True)

# Wind data
cardinal_degree = {
        'N'   : [348.75 ,  11.25],
        'NNE' : [11.25  ,  33.75],
        'NE'  : [33.75  ,  56.25],
        'ENE' : [56.25  ,  78.75],
        'E'   : [78.75  , 101.25],
        'ESE' : [101.25 , 123.75],
        'SE'  : [123.75 , 146.25],
        'SSE' : [146.25 , 168.75],
        'S'   : [168.75 , 191.25],
        'SSW' : [191.25 , 213.75],
        'SW'  : [213.75 , 236.25],
        'WSW' : [236.25 , 258.75],
        'W'   : [258.75 , 281.25],
        'WNW' : [281.25 , 303.75],
        'NW'  : [303.75 , 326.25],
        'NNW' : [326.25 , 348.75]
}

def wind_dir(data):
    '''
    Dataset with zonal and meridional coordinates. Outputs the cardinal- and 
    the degree direction, respectively.
    '''
    car=[]
    deg=[]
    u,v = data['zonal'], data['meridional']
    wind_degree = 180/np.pi*np.arctan2(-u,-v)+180

    for ind, bear in enumerate(wind_degree):
        for direction, interval in cardinal_degree.items():
            low, high = interval    
            if bear >= low and bear < high:
                car.append(direction)
                deg.append(bear)
        if ind != len(car)-1:
            car.append('N')
            deg.append(bear)   
    return car, deg


data['car_dir'], data['deg_dir'] = wind_dir(data)
weather_forecast['car_dir'], weather_forecast['deg_dir'] =\
                                        wind_dir(weather_forecast)


# Defining the regression models 

def linear_regression(X_train, y_train, X_test, y_test, plot=True):
    '''
    Linear regressor function that trains the model, predicts, prints 
    errors and plots the results. 
    '''
    regr = LinearRegression()
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # Error
    rms_r2_score(y_test, y_pred)
    
    if plot:
        # Plot outputs
        #plot_task1(X_test, y_test, y_pred, 'Linear Regression')
        plot_powergeneration(y_test, y_pred, 'Linear Regression')
    return y_pred        


def knn_crossval(X,y,n_folds=10):
    '''
    Defining hyperparameters for grid search, and performs 
    cross validation.
    '''
    num_neighbors = [1, 5, 20, 50, 100, 500, 800, 1000]
    param_grid = [{'n_neighbors': num_neighbors,
                   'weights': ['uniform']},
                 {'n_neighbors': num_neighbors,
                  'weights': ['distance']}]
    grid_search = GridSearchCV(KNeighborsRegressor(),
                               param_grid,
                               cv=n_folds,
                               n_jobs=-1,
                               scoring='neg_mean_squared_error',
                               verbose=10)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    return grid_search.best_params_    


def k_nearest_neighbors(X_train, y_train, X_test, y_test,params=None, 
                        plot=True, gridsearch=False):
    '''
    K-nearest neighbors function that trains the model, predicts, prints 
    errors and plots the results. Cross validation and grid search for 
    hyperparameters are used to get the best model.
    '''

    if gridsearch:
        params = knn_crossval(X_train, y_train)
    neigh = KNeighborsRegressor().set_params(**params)
    
    neigh.fit(X_train_selected, y_train) 
    y_pred = neigh.predict(X_test)

    # The Root mean squared error
    print("Root Mean squared error: %.4f"
          % np.sqrt(mean_squared_error(y_test, y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    
    if plot:
        # Plot outputs
        #plot_task1(X_test, y_test, y_pred, 'KNN')
        plot_powergeneration(y_test, y_pred, 'KNN')
    return y_pred


def svr_crossval(X, y, n_folds=10):
    '''
    Defining hyperparameters for grid search, and performs 
    cross validation.
    '''
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1, 10]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVR(kernel='rbf'),
                               param_grid,
                               cv=n_folds,
                               n_jobs=-1,
                               verbose=10)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    return grid_search.best_params_        
        

def support_vector_regression(X_train, y_train, X_test, y_test,params=None, 
                              plot=True, gridsearch=False):
    '''
    SVR function that trains the model, predicts, prints 
    errors and plots the results. Cross validation and grid search for 
    hyperparameters are used to get the best model.
    '''

    if gridsearch:
        params = svr_crossval(X_train, y_train)
    svr_rbf = SVR().set_params(**params)
    
    y_pred = svr_rbf.fit(X_train_selected, y_train).predict(X_test_selected)

    # The Root mean squared error
    print("Root Mean squared error: %.4f"
          % np.sqrt(mean_squared_error(y_test, y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))
    
    if plot:
        #plot_task1(X_test, y_test, y_pred, 'SVR')
        plot_powergeneration(y_test, y_pred, 'SVR')
    return y_pred


def ann_model(X_train, y_train, X_test, y_test, plot=True):
    '''
    Trains an artificial neural network. n-input channels and one 
    output(the predicted power).
    '''
    input_shape = X_train.shape[1]
    output_shape = 1
    model = Sequential()
    # Building the neural network
    model.add(InputLayer(input_shape=(input_shape,)))
    model.add(Dense(30, kernel_initializer='lecun_normal',
                    bias_initializer='ones', activation='selu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, kernel_initializer='lecun_normal',
                    bias_initializer='ones',activation='softmax'))
    model.add(Dense(output_shape))

    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=10,  verbose=0)

    y_pred = model.predict(X_test)
    
    # Error
    rms_r2_score(y_test, y_pred)

    if plot:
        #plot_task1(X_test, y_test, y_pred, 'ANN')
        plot_powergeneration(y_test, y_pred, model='ANN')
    return y_pred


def rms_r2_score(y_test, y_pred):
    # The Root mean squared error
    print("Root Mean squared error: %.4f"
          % np.sqrt(mean_squared_error(y_test, y_pred)))
    # Explained variance score: 1 is perfect prediction
    print('R^2: %.2f' % r2_score(y_test, y_pred))


# Store predictions to file function
def store_predictions_to_file(y_pred, model=None, task=1,
                              template='ForecastTemplate.csv'):
    pred = pd.read_csv(template)
    pred['FORECAST'] = y_pred[:len(pred)]
    pred.to_csv('ForecastTemplate{1}-{0}.csv'.format(model, task), index=False)


# Plotting function

def plot_powergeneration(y_test, y_pred, model=None):
    plt.figure(figsize=(15, 5))

    plt.plot(y_test.values, color='darkorange', label='Real')

    plt.plot(y_pred, color='navy', label='Predicted')

    plt.xlabel('Time')
    plt.ylabel('Wind Power')
    plt.title(model)
    plt.legend()
    # plt.ylim(-0.1,y_test.max().all()+0.1)
    plt.show()


def plot_task1(X_test, y_test, y_pred, model=None):
    plt.scatter(X_test, y_test, color='darkorange',
                marker='.', label='Real', linewidth=0.1)

    plt.scatter(X_test, y_pred, color='navy',
                marker='.', label='Predicted', linewidth=0.1)

    plt.xlabel('Wind speed')
    plt.ylabel('Wind Power')
    plt.title(model)
    plt.legend()
    plt.ylim(-0.1, y_test.max().all() + 0.1)
    plt.show()



# Task 1

#Select training dataset

train = data.drop('POWER', 1)
train_y = data['POWER']#.reshape(-1,1)

#X_train, X_test, y_train, y_test =\
#        train_test_split(train,train_y,test_size=0.3,random_state=1)
sel=-1               
X_train = train[:sel]
y_train = train_y[:sel]

X_test = weather_forecast
y_test = solution
            
X_train_selected = X_train['windspeed'].values.reshape(-1, 1)
X_test_selected = X_test['windspeed'].values.reshape(-1, 1)
            
print(X_train_selected.shape,
      X_test_selected.shape, 
      y_train.shape, y_test.shape)


# Training and predicting
train_test = X_train_selected, y_train, X_test_selected, y_test

np.random.seed(1)
linear_regression(*train_test)
k_nearest_neighbors(*train_test, params={'n_neighbors': 800})
y_pred=support_vector_regression(*train_test, {'C': 0.1, 'gamma': 0.01})
pred=ann_model(*train_test)


# TASK 2
# Define dataset
train = data.drop('POWER', 1)
train_y = data['POWER']#.reshape(-1,1)

#X_train, X_test, y_train, y_test =\
#        train_test_split(train,train_y,test_size=0.3,random_state=1)
                        
X_train = train
y_train = train_y

X_test = weather_forecast
y_test = solution

X_train_one = X_train['windspeed'].values.reshape(-1 ,1)
X_test_one = X_test['windspeed'].values.reshape(-1 ,1)

X_train_two = X_train[['windspeed','deg_dir']]
X_test_two = X_test[['windspeed','deg_dir']]

#Training and prediction
train_test_two = X_train_two, y_train, X_test_two, y_test
train_test_one = X_train_one, y_train, X_test_one, y_test

lr_pred=linear_regression(*train_test_one,plot=False)
mlr_pred=linear_regression(*train_test_two,plot=False)


# Plotting MLR, LR and real values
plt.figure(figsize=(15,5))

plt.plot(y_test.values, color='darkorange', label='Real')

plt.plot(mlr_pred, color='navy', label='MLR_Predicted')
plt.plot(lr_pred, color='darkgreen', label='LR_Predicted')

plt.xlabel('Time')
plt.ylabel('Wind Power')
plt.title('MLR and LR')
plt.legend()
#plt.ylim(-0.1,y_test.max().all()+0.1)
plt.show()

#TASK 3

#Linear Regression
train_y = data['POWER']
train = np.arange(len(train_y)).reshape(-1,1)
y_test = solution
test = np.arange(train[-1],len(y_test)+train[-1]).reshape(-1, 1)

train_test = train, train_y, test, y_test

pred=linear_regression(*train_test, plot=False )

#RNN

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    '''
    Transformes the timeseries data into supervised dataset
    
                             [a1, a2]
                             [a2, a3]
    [a1,a2,a3,a4,a5,a6] ---> [a3, a4]
                             [a4, a5]
                             [a5, a6]
                             [a6,  0]
    '''
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
 
# fit an rnn network to training data
def fit_rnn(train, batch_size, nb_epoch, neurons):
    '''
    Defines the RNN model and fits to the training data.
    '''
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(SimpleRNN(neurons, batch_input_shape=(batch_size, 
                                                    X.shape[1], 
                                                    X.shape[2]), 
                        stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001))
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, 
                  shuffle=False)
        model.reset_states()
        if i%10==0:
                print('Finished with %s epochs' %(i+1))
    return model
 
# make a one-step forecast
def forecast_rnn(model, batch_size, X):
    '''
    Uses a pre-trained model and predicts the next value.
    '''
    X = X.reshape(len(X),1, 1 )
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

# transform data to be stationary
series = train_y
raw_values = series.values[80:]#[-3000:]

# transform data to be supervised learning
supervised = timeseries_to_supervised(raw_values, 1)

supervised_values = supervised.values
 
batch_size = 100
epochs = 100
neurons = 4
print('**----Start train model----**')
# fit the model
rnn_model = fit_rnn(supervised_values, batch_size, epochs, neurons)
print('**----Finished training----**')

# forecast the entire training dataset to build up state for forecasting
train_reshaped = supervised_values[:, 0].reshape(len(supervised_values), 1, 1)
rnn_model.predict(train_reshaped, batch_size=batch_size)

# walk-forward validation on the test data
predictions = list()
history = supervised_values[-batch_size:, -2]
for i in range(len(y_test)):
    X = np.array(history[-batch_size:])
    yhat = forecast_rnn(rnn_model, batch_size, X)
    # store forecast
    predictions.append(yhat)
    history = np.concatenate((history, np.array([yhat])), axis=0)
    
# report performance
r2 = r2_score(y_test.values, predictions)
print('R^2: %.4f' % r2)
rmse = np.sqrt(mean_squared_error(y_test.values, predictions))
print('RMSE: %.4f' % rmse)

# line plot of observed vs predicted
#plot data
fig, ax = plt.subplots(figsize=(15,7))
#data.plot(ax=ax)
ax.plot(y_test.index,y_test.values, label='Real')
#set ticks every week
ax.xaxis.set_major_locator(DayLocator())
#set major ticks format
ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
ax.plot(pd.DataFrame(predictions,index=y_test.index),label='RNN')
ax.plot(pd.DataFrame(pred,index=y_test.index),label='LR')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()

# The function used to manually store the predictions.
store_predictions_to_file(predictions, 'RNN', task=3)
