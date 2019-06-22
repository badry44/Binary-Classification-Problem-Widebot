import pandas as pd
import numpy as np
np.random.seed(1337)
import random
random.seed(1337)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import set_random_seed
set_random_seed(1337)
import random

train_data = pd.read_csv('training.csv',sep = ';')
validation_data =  pd.read_csv('validation.csv',sep = ';')

def refactorDataSet(train_data , meanCalculated = False,var21Med=0,var22Med=0,var14Mean=0) :
    # variable 1 edit 
    train_data['variable1']=np.where(train_data['variable1']=="a", 0, 1)
    #var1Mean = train_data['variable1'].mean()
    #if (var1Mean<=0.5):
    #    var1ReplacedNan = 0
    #else :
    #    var1ReplacedNan = 1
    var1ReplacedNan = 1
    train_data['variable1']=np.where(train_data['variable1'].isnull(), var1ReplacedNan, train_data['variable1'])
    train_data['variable1']=train_data['variable1'].astype('int')
    #################

    # variable 2 edit 
    train_data["variable2"]= train_data["variable2"].astype(str) 
    # split var2 into 2 var and make it string
    train_data['variable2-1'] = np.where(np.logical_and(train_data['variable2']!= 'None' , train_data['variable2'].astype(str)!='nan'),train_data['variable2'].astype(str).str.split(",", expand = True)[0] ,"0")
    train_data['variable2-2'] = np.where(np.logical_and(train_data['variable2']!= 'None' , train_data['variable2'].astype(str)!='nan'),train_data['variable2'].astype(str).str.split(",", expand = True)[1] ,"0")
    #remove None
    train_data = train_data[train_data['variable2-1'].isnull()==False]
    train_data = train_data[train_data['variable2-2'].isnull()==False]
    train_data["variable2-1"] = train_data["variable2-1"].apply(int)
    train_data["variable2-2"] = train_data["variable2-2"].apply(int)
    train_data.drop('variable2', axis=1, inplace=True)
    if (meanCalculated==False) :
        var21Med = train_data['variable2-1'].median() #28
        var22Med = train_data['variable2-2'].median() #42
    train_data['variable2-1'] = np.where(train_data['variable2-1']==0,var21Med,train_data['variable2-1'])
    train_data['variable2-2'] = np.where(train_data['variable2-2']==0,var22Med,train_data['variable2-2'])
    #################

    # var 3 edit 
    train_data["variable3"] = train_data["variable3"].str.replace(',','.').apply(float)
    ####################

    # varaible4 edit
    # u =2800 , y =464 , l =32
    train_data['variable4']=np.where(train_data['variable4'].isnull(), 'u', train_data['variable4'])
    train_data['variable4']=np.where(train_data['variable4']=='u', 2, train_data['variable4'])
    train_data['variable4']=np.where(train_data['variable4']=='y', 1, train_data['variable4'])
    train_data['variable4']=np.where(train_data['variable4']=='l', 0, train_data['variable4'])
    train_data['variable4']=train_data['variable4'].astype('int')
    ###########

    # varaible5 edit
    # g =2800 , p =464 , gg =32
    #train_data['variable5']=np.where(train_data['variable5'].isnull(), 'g', train_data['variable5'])
    #train_data['variable5']=np.where(train_data['variable5']=='g', 2, train_data['variable5'])
    #train_data['variable5']=np.where(train_data['variable5']=='p', 1, train_data['variable5'])
    #train_data['variable5']=np.where(train_data['variable5']=='gg  ', 0, train_data['variable5'])
    train_data.drop('variable5', axis=1, inplace=True)
    ###########


    #varaible6 edit
    #c-q-W-x-cc-aa-m-i-k-ff-e-d-r-j
    train_data['variable6']=np.where(train_data['variable6'].isnull(),'c' , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='j',0 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='r',1 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='d',2 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='e',3 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='ff',4 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='k',5 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='i',6 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='m',7 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='aa',8 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='cc',9 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='x',10 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='W',11 , train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='q',12, train_data['variable6'])
    train_data['variable6']=np.where(train_data['variable6']=='c',13, train_data['variable6'])
    train_data['variable6']=train_data['variable6'].astype('int')
    
    ###########


    # variable7 edit 

    #v= 1905
    train_data['variable7']=np.where(train_data['variable7'].isnull(),'v' , train_data['variable7'])
    train_data['variable7']=np.where(train_data['variable7']=='n',0 , train_data['variable7'])
    train_data['variable7']=np.where(train_data['variable7']=='o',1 , train_data['variable7'])
    train_data['variable7']=np.where(train_data['variable7']=='j',2 , train_data['variable7'])
    train_data['variable7']=np.where(train_data['variable7']=='dd',3 , train_data['variable7'])
    train_data['variable7']=np.where(train_data['variable7']=='z',4 , train_data['variable7'])
    train_data['variable7']=np.where(train_data['variable7']=='ff',5 , train_data['variable7'])
    train_data['variable7']=np.where(train_data['variable7']=='bb',6 , train_data['variable7'])
    train_data['variable7']=np.where(train_data['variable7']=='h',7 , train_data['variable7'])
    train_data['variable7']=np.where(train_data['variable7']=='v',8 , train_data['variable7'])
    train_data['variable7']=train_data['variable7'].astype('int')


    #####################

    # variable8 edit 


    train_data["variable8"]=train_data["variable8"].str.replace(',','.').apply(float)

    ######################

    # variable9 edit 

    train_data['variable9']=np.where(train_data['variable9']=='f',0 , train_data['variable9'])
    train_data['variable9']=np.where(train_data['variable9']=='t',1 , train_data['variable9'])
    train_data['variable9']=train_data['variable9'].astype('int')

    #######################

    #variable10 edit


    train_data['variable10']=np.where(train_data['variable10']=='f',0 , train_data['variable10'])
    train_data['variable10']=np.where(train_data['variable10']=='t',1 , train_data['variable10'])
    train_data['variable10']=train_data['variable10'].astype('int')
    ##################

    #variable11 edit


    ##################

    #variable12 edit
    train_data['variable12']=np.where(train_data['variable12']=='f',0 , train_data['variable12'])
    train_data['variable12']=np.where(train_data['variable12']=='t',1 , train_data['variable12'])
    train_data['variable12']=train_data['variable12'].astype('int')

    #################
    #variable13 edit
    #p>s>g
    train_data['variable13']=np.where(train_data['variable13']=='p',0 , train_data['variable13'])
    train_data['variable13']=np.where(train_data['variable13']=='s',1 , train_data['variable13'])
    train_data['variable13']=np.where(train_data['variable13']=='g',2 , train_data['variable13'])
    train_data['variable13']=train_data['variable13'].astype('int')

    ################

    #varaible14 edit
    if (meanCalculated==False) :
        var14Mean = np.floor(train_data['variable14'].mean())#159
    train_data['variable14']=np.where(train_data['variable14'].isnull(),var14Mean , train_data['variable14'])

    ###############

    #variable17 edit
    #var17Mean = np.floor(train_data['variable17'].mean())
    #print ("var17Mean : " ,var17Mean) #1598825
    #train_data['variable17']=np.where(train_data['variable17'].isnull(),var17Mean , train_data['variable17'])
    train_data.drop('variable17', axis=1, inplace=True)

    ###############

    #variable18 and 19 edit

    train_data.drop('variable18', axis=1, inplace=True)
    train_data.drop('variable19', axis=1, inplace=True)
    train_data.drop('variable4', axis=1, inplace=True)
    #important 9-
    ###############
    
    #classLabel
    train_data['classLabel']=np.where(train_data['classLabel']=='no.',0 , train_data['classLabel'])
    train_data['classLabel']=np.where(train_data['classLabel']=='yes.',1 , train_data['classLabel'])
    train_data['classLabel']=train_data['classLabel'].astype('int')
    
    #train_data.info()
    
    ###############
    return train_data,var21Med,var22Med,var14Mean
    
    
    

#Sandarize all the data 
#here i reached 72% accurcy in validation data
def UsingLogisticV1(train_data,Y_train,validation_data,Y_validation) :
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(train_data, Y_train)
    #pipe.predict_proba(validation_data)
    print('Accuracy of logistic regression classifier  on validation set: {:.2f}'.format(pipe.score(validation_data, Y_validation)))
    print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(pipe.score(train_data, Y_train)))
    
    

#here i reached 65% on validation data
def UsingLogisticV2(train_data,Y_train,validation_data,Y_validation) :
    logreg = LogisticRegression()
    logreg.fit(train_data, Y_train)
    NomalizedVairables = ['variable2-2','variable2-1','variable15','variable14','variable3']
    train_data = NormalizedTheData(train_data,NomalizedVairables)
    validation_data = NormalizedTheData(validation_data,NomalizedVairables)
    y_pred = logreg.predict(validation_data)
    print('Accuracy of logistic regression classifier on validation set: {:.2f}'.format(logreg.score(validation_data, Y_validation)))
    print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logreg.score(train_data, Y_train)))
    
    

# here I reached 0.80 accurcy in validation data
# with normalzation to variable2-2-variable2-1-variable15-variable14-variable3 i got 77% and with all varaibles i had reached  80% accuracy on validation data
def DNNWithTensorFlow(train_data,Y_train,validation_data,Y_validation):
    model = keras.Sequential([
    keras.layers.Dense(128,activation=tf.nn.relu,input_shape=(14,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)])
    NomalizedVairables = ['variable2-2','variable2-1','variable15','variable14','variable3','variable1','variable6','variable7','variable8','variable9','variable10','variable13','variable12']
    train_data = NormalizedTheData(train_data,NomalizedVairables)
    validation_data = NormalizedTheData(validation_data,NomalizedVairables)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, Y_train, epochs=100)
    results = model.evaluate(validation_data, Y_validation)
    print('test loss, test acc:', results)

def NormalizedTheData (train_data,NomalizedVairables) :
    #Normalized this variable2-2-variable2-1-variable15-variable14-variable3
    for var in NomalizedVairables : 
        train_data[var]=(train_data[var]-train_data[var].mean())/train_data[var].std()
    return train_data

train_data,vari21Med,vari22Med,vari14Mean = refactorDataSet(train_data)
Y_train = train_data["classLabel"]
train_data.drop('classLabel', axis=1, inplace=True)
validation_data,vari21Med,vari22Med,vari14Mean = refactorDataSet(validation_data,True,vari21Med,vari22Med,vari14Mean)
Y_validation = validation_data["classLabel"]
validation_data.drop('classLabel', axis=1, inplace=True)

#UsingLogisticV1(train_data,Y_train,validation_data,Y_validation)
#UsingLogisticV2(train_data,Y_train,validation_data,Y_validation)
DNNWithTensorFlow(train_data,Y_train,validation_data,Y_validation)


# data preprocessing notes   
# 1 - discard variable 18 cause it have alot of Nan
# 2 - split variable 2 into 2 variables and 3 is double 
# 3 - overfiting on varaible 19 so i wii  remove it from dataset
# 4 - variable5 is redundant  out of variable4 so i will remove it
# 5- there is a relation between var10 and var11 as if var10 = f then var11 = 0 else var 10 will be numerical number 
# 6- var17 = var14 *1000 so var17 is redundant var
# 7 - with the above changes i got with logistc regression acc on validation set = 0.68 "without normalzation"
































#print(train_data.head())
#train_data.info()
#print('_'*40)
#validation_data.info()
#train_data.to_excel("training.xlsx")
#print(train_data[(train_data['variable1']=="a")])
#print(train_data[(train_data['variable1']=="b")])

# get most common colounm by print (train_data['variable4'].value_counts())

#print(train_data.shape)
#print(len(Y_train))
#print(validation_data.shape)
#print(len(Y_validation))
# print(train_data['variable1'].unique())
# print ("variable1 end")
# print(train_data['variable2-1'].unique())
# print ("variable2-1 end")
# print(train_data['variable2-2'].unique())
# print ("variable2-2 end")
# print(train_data['variable3'].unique())
# print ("variable3 end")
# print(train_data['variable4'].unique())
# print ("variable4 end")
# print(train_data['variable6'].unique())
# print ("variable6 end")
# print(train_data['variable7'].unique())
# print ("variable7 end")
# print(train_data['variable8'].unique())
# print ("variable8 end")
# print(train_data['variable9'].unique())
# print ("variable9 end")
# print(train_data['variable10'].unique())
# print ("variable10 end")
# print(train_data['variable11'].head())
# print ("variable11 end")
# print(train_data['variable12'].unique())
# print ("variable12 end")
# print(train_data['variable13'].unique())
# print ("variable13 end")
# print(train_data['variable14'].head())
# print ("variable14 end")
# print(train_data['variable15'].head())
# print ("variable15 end")
#print(train_data['variable17'].unique())
#print ("variable17 end")
#print(train_data['variable18'].head())
#print(train_data['variable19'].head())
#print ("variable19 end")























