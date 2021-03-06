# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
import time
import json
from tqdm import tqdm, tqdm_notebook
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import matplotlib.pyplot as plt
from keras.applications.xception import Xception
from utils.losses import categorical_crossentropy
from utils.Mertics import f1
from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger
from models.Xception_Classification import XceptionClass
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from utils.CallBacks import *
import utils.config
# Any results you write to the current directory are saved as output.
from utils.ImageAugementation import function_line
from utils.config import config
from utils.config import config

modeldict=config.modeldict



train_images_dir=config.train_images_dir

dataset_dir=config.dataset_dir

root_dir=config.root_dir

train_csv_path=config.train_csv_path
print(train_csv_path)

test_images_dir=config.test_images_dir

test_csv_path=config.test_csv_path

num_gpu=config.num_gpu

droprate=config.droprate

kernel_re=config.kernel_re

img_h,img_w=config.img_h,config.img_w

batch_size=config.batch_size

modelname=config.modelname

weightsname=config.weightsname

nb_classes =config.nb_classes

lr=config.lr

tensorboarddir='logs'
modeldir=os.path.join(root_dir,'modelinfo',modelname)
if not os.path.exists(root_dir+'/modelinfo'):
    os.mkdir(root_dir+'/modelinfo')
if not os.path.exists(modeldir):
    os.mkdir(modeldir)
if not os.path.exists(modeldir+'/'+'weights'):
    os.mkdir(modeldir+'/'+'weights')
if not os.path.exists(modeldir+'/'+tensorboarddir):
    os.mkdir(modeldir+'/'+tensorboarddir)


train_l=os.listdir(train_images_dir)
print('train length',len(train_l))
train_df = pd.read_csv(train_csv_path,dtype = {'category_id': str})
print(train_df.shape)

test_l=os.listdir(test_images_dir)
print(len(test_l))
test_df = pd.read_csv(test_csv_path)
print(test_df.shape)
train_df.describe()
train_df['category_id'] = train_df['category_id'].astype(str)
h=train_df['category_id'].value_counts()

print(h.dtypes)

# h.plot(kind='bar')

test_df.head()
print test_df.describe()
train_df.head()
print train_df.describe()



datagen = ImageDataGenerator(
#     zca_whitening=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=1.0/255.0,
    preprocessing_function=None,
    validation_split=0.1)

train_gen=datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = train_images_dir,
        x_col = 'file_name', y_col = 'category_id',
        target_size=(img_h,img_w),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')
val_gen=datagen.flow_from_dataframe(
        dataframe = train_df,
        directory = train_images_dir,
        x_col = 'file_name', y_col = 'category_id',
        target_size=(img_h,img_w),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')


print(set(train_gen.class_indices))

if modelname not in modeldict.keys():
    print('---The '+modelname+'have not defined in modeldict ---')
    os._exists(0)

model=modeldict[modelname](input_shape=(img_h,img_w,3),classes=nb_classes,droprate=droprate,kernel_regu_rate=kernel_re)

# if num_gpu>1:
#     model=multi_gpu_model(model,num_gpu)

sgd=SGD(lr=lr,decay=1e-3,momentum=0.9,nesterov=True)

adam=Adam(lr=lr,decay=0.0001)

modelcheck=ModelCheckpoint(filepath=modeldir+'/'+'weights'+'/{epoch:02d}-{val_loss:.2f}-{loss:.2f}.h5',monitor='val_loss',save_weights_only=True)
csvlog=CSVLogger(filename=modeldir+'/csv_path.csv',separator=',',append=True)
earstop=EarlyStopping(patience=15,monitor='val_loss')



model.summary()


model.compile(loss=categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy',f1])



# Train model
nb_epochs=config.nb_epochs

if os.path.exists(modeldir+'/'+'weights'+'/'+str(weightsname)) and (not weightsname==None):
    model.load_weights(modeldir+'/'+'weights'+'/'+weightsname)
    print('---model weights load successfull---')
    print('---'+modeldir+'/'+'weights'+'/'+weightsname+'---')
else:
    print('----------not load weihts!!!!!!!!!-----------')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
        dataframe = test_df,
        directory = test_images_dir,
        x_col = 'file_name', y_col = None,
        target_size = (img_h,img_w),
        batch_size = 10,
        shuffle = False,
        class_mode = None
        )

test_generator.reset()
predict = model.predict_generator(test_generator, steps = len(test_generator.filenames)/10,verbose=1)
predicted_class_indices=np.argmax(predict,axis=1)
labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
sam_sub_df = pd.read_csv(dataset_dir+'/sample_submission.csv')
print(sam_sub_df.shape)
sam_sub_df.head()
filenames=test_generator.filenames
results=pd.DataFrame({"Id":filenames,
                      "Predicted":predictions})
s=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
results['Id'] = results['Id'].map(lambda x: str(x)[:-4])
results.to_csv(model_dir+'/'+s+"-results.csv",index=False)