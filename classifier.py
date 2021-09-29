import numpy as np
import pandas as pd 
import csv
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']

print(pd.Series(y).value_counts())

classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

nclasses = len(classes)


x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=9,train_size=7500,test_size=2500)
x_train_scale = x_train/255
x_test_scale = x_test/255

clf = LogisticRegression(solver='saga',multi_class='multinomial').fit(x_train_scale,y_train)

def get_pre(image):
    im_pil = Image.open(image)
    image_w = im_pil.convert('l')
    image_w_resize = image_w.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_w_resize,pixel_filter)
    image_w_resize_inverted_sc = np.clip(image_w_resize-min_pixel,0,255)
    max_pixel = np.max(image_w_resize)
    image_w_resize_inverted_sc = np.asarray(image_w_resize_inverted_sc)/max_pixel
    test_sample = np.array(image_w_resize_inverted_sc).reshape(1,784)
    test_prd = clf.predict(test_sample)
    return test_prd[0]