
from keras.callbacks import  BaseLogger,ReduceLROnPlateau


import numpy as np
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping,TensorBoard


lrreduce=ReduceLROnPlateau(monitor='loss',factor=0.2,patience=3,min_lr=0.00001)

# TODO
# need finish the callback of print lr

