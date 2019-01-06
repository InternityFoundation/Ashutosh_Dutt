'''OM SHRI GANESHAY NAMAH'''
''' loading the data using pandas'''
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae# : for evaluation of maodel
data= pd.read_csv('melb_data.csv')
#print(data.describe()) to just get a confirmation and overview of loaded data
#print(data.columns) : to get look on columns in data set
data= data.dropna(axis=0) #:to drop missing values


"""choosing features for predictions"""
y= data.Price  #: choosing output features
input_feature=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X= data[input_feature]
#print(X.head()): to check if data is loaded or not


'''Splitting the data to get more accurate model'''
from sklearn.model_selection import train_test_split as tts
train_X,test_X,train_y,test_y= tts(X,y,random_state=0)


''' Selecting a model for prediction'''
def dctModel(train_X,test_X,train_y,test_y):
    '''here we choose decision tree regressor for the prediction '''
    from sklearn.tree import DecisionTreeRegressor as dct
    model= dct(random_state= 1,max_leaf_nodes=500)# define the model
    model.fit(train_X,train_y) # fit the model
    pred_y= model.predict(test_X) #make the model predict 
    '''Evaluating your model using mean absolute error'''
    error = mae(test_y,pred_y)
    return(error)


print("Error in decision tree model is %f :",dctModel(train_X,test_X,train_y,test_y))

    