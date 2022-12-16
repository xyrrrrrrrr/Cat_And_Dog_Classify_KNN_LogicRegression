'''Logic_Regression.py

This module is for classification using logic regression.

@author: Kainan Chen
@date: 2022/12/16
@version: 1.0.0 
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.linear_model import LogisticRegression
import pickle

data_dir = './rawdata_25%/train/'

# use cat and dog pictures as data
# cat: 0, dog: 1
# load data
def resize_data(data):
    data = Image.fromarray(data)
    data = data.resize((100, 100)).convert('L')
    # convert to 1d array
    data = np.array(data).reshape(-1)
    # formalize data
    data = np.array(data) / 255
    return data


def load_data():
    print('loading data...')
    cat = os.listdir(data_dir + 'cat')
    dog = os.listdir(data_dir + 'dog')
    cat = [np.array(Image.open(data_dir + 'cat/' + i)) for i in cat]
    dog = [np.array(Image.open(data_dir + 'dog/' + i)) for i in dog]
    for i in range(len(cat)):
        cat[i] = resize_data(cat[i])
    for i in range(len(dog)):
        dog[i] = resize_data(dog[i])
    cat = np.array(cat)
    dog = np.array(dog)
    data = np.concatenate((cat, dog))
    label = np.concatenate((np.zeros(len(cat)), np.ones(len(dog))))
    return data, label

# train model
def train_model(data, label):
    model = LogisticRegression()
    print('training model...')
    model.fit(data, label)
    with open('Logic_Regression.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

# test model
def test_model(model, data, label):
    print('testing model...')
    pred = model.predict(data)
    acc = np.sum(pred == label) / len(label)
    return acc

if __name__ == '__main__':
    data, label = load_data()
    model = train_model(data, label)
    acc = test_model(model, data, label)
    print('accuracy: {}'.format(acc))