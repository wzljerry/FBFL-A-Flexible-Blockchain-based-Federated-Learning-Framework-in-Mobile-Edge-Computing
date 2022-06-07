# -*- coding: utf-8 -*-
'''
@Title : Federated Learning From Scratch
@Author : Zhilin Wang, Shengyang Li
@Email : {wangzhil,sl137@iu.edu}@iu.edu
@Date : 14-04-2022
@Reference: https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399
'''
# import the packages required
import tensorflow as tf
import hashlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.datasets import mnist
import keras
import random
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as Ks
from random import choice

# to get the training data, and split the data via the number of clients
class Get_data:
  def __init__(self,n):
    self.n=n # number of clients

  # download data
  def load_data(self):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test

  # split the data according to the number of clients
  def split_data(self, data, n): 
    size=int(len(data) / self.n)
    s_data = []
    for i in range(0, int(len(data)) + 1, size):
        c_data = data[i:i + size]
        if c_data != []:
            s_data.append(c_data)
    return s_data
  
  # data preparation
  def pre_data(self):

    X_train, y_train, X_test, y_test=self.load_data()
    X_train=self.split_data(X_train,self.n) 
    y_train=self.split_data(y_train,self.n)

    return X_train, y_train, X_test, y_test

  # data type
  def iid_data(self,data,label):

    index=[i for i in range(len(data))]
    random.shuffle(index)
    data=data[index]
    label=label[index]

    return data,label

# basic model
class Model:
  def __init__(self):
    pass

  # global model
  def global_models(self):

    model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(10, activation='softmax')
      ])

    return model
  # evaluate the global model
  def evaluate_model(self,model,test_X, test_y):

    model.compile(optimizer='adam',
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])
    score=model.evaluate(test_X, test_y)
    print("Test Loss:", score[0])
    print('Test Accuracy:', score[1])

    return score[0],score[1]

# clients
class Client:

  def __init__(self,n):
    self.n=n
    self.lr = 0.01 
    self.loss='categorical_crossentropy'
    self.metrics = ['accuracy']
    self.optimizer = SGD(lr=self.lr, 
                decay=self.lr / 2, 
                momentum=0.9
               )

  # get the weight for each update
  def weight_client(self,data,m):
    wei_client = []
    for i in range(self.n):
        len_data = len(data[i])
        proba = len_data / m
        wei_client.append(proba)

    return wei_client
  
  # weights time the updates
  def scale_model_weights(self,weight,scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
      weight_final.append(scalar[i]*weight[i])

    return weight_final


  def training(self,X,y,global_weights,factor):

    #fact=self.weight_client()
    model=Model().global_models()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    model.set_weights(global_weights)
    model.fit(X,y,epochs=1)

    weights = self.scale_model_weights(model.get_weights(), factor)
    print('========================================================================')
    return weights

  #attack models
  def model_poisioning_attack(self,i,weight):
    weigh=[]
    if (i==1) or (i ==2) :
        steps = len(weix)
        scalar=[2 for i in range(steps)]
        for j in range(steps):
          weigh.append(scalar[j]*weix[j]) # change the updates

    return weigh

  def data_poisoning_attack(self,i,data):

    if i==1&i==2:
      data[i]=data[i]+1 # change the data
    return data

  def label_flipping_atack(self,label):

    for i in label:
      if i == 1:
        i==9 # flip the label

    return label

# server
class Server:

  def __init__(self):
    pass

  def sum_scaled_weights(self,scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad
    #evaluate global model
  def evaluate(self,model,test_X, test_y): 
    loss,acc=Model.evaluate_model(model,test_X, test_y)

    return loss,acc

def main(num_client,K,client_num):

  #data preperation
  client=Client(num_client)
  get_data=Get_data(num_client)
  train_X,train_y, test_X, test_y=get_data.load_data()
  m=len(train_X)

  #non-iid
  #train_X,train_y=iid_data(train_X,train_y)

  #print(len(train_y))
  X_train, y_train, X_test, y_test=get_data.pre_data()

  server=Server()
  global_model=Model().global_models()

  accuracy=[]
  losses=[]
  factor=client.weight_client(X_train,m)

  # k rounds
  for k in range(K):
    global_weights=global_model.get_weights()
    weit=[]
    for i in range(client_num):
      # updates of each client
      weix=client.training(X_train[i], y_train[i],global_weights,factor)
      
      weit.append(weix)

      Ks.clear_session()

    # global model updates
    global_weight=server.sum_scaled_weights(weit) # fedavg
    # set the global model
    global_model.set_weights(global_weight)
    print('============================== Training Done ==============================')
    # accuracy and loss
    loss,acc=Model().evaluate_model(global_model,test_X,test_y)
    losses.append(loss)
    accuracy.append(acc)

  # will return two list
  return losses,accuracy

def vitialization(k,loss,acc,label,x_label,y_label,file_name):

  for i in range(k):
    plt.plot(k,acc[i],label=label[i])
    #plt.plot(k,loss[i],label='IID')
    plt.xlabel(x_label,fontsize=20)
    plt.ylabel(y_label,fontsize=20)
    #plt.ylim(18000,32000)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.legend(fontsize=15,framealpha=0)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

# this is for single round
def client_train(num_client,K,client_num):

  client=Client(num_client)
  get_data=Get_data(num_client)
  train_X,train_y, test_X, test_y=get_data.load_data()
  X_train, y_train, X_test, y_test=get_data.pre_data()
  m=len(train_X)

  server=Server()
  global_model=Model().global_models()
  accuracy=[]
  losses=[]
  factor=client.weight_client(X_train,m)
  global_weights=global_model.get_weights()
  weit=[]

  for i in range(client_num):
    weix=client.training(X_train[i], y_train[i],global_weights,factor)
    weit.append(weix)
    Ks.clear_session()

  return weit


if __name__=='__main__':
  # basic parameters
  num_client=10
  K=50
  client_num=5
  ac=[]
  lo=[]
  label=[]
  x_label="The Number of Episode"
  y_label='Accuracy'
  file_name="test.csv"

  loss,acc=main(num_client,K,client_num)

  k=np.arange(1,K+1,1)
  
  # plot diagrams
  #vitialization(k,loss,acc,label,x_label,y_label,file_name)
