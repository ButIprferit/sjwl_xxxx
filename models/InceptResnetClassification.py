from keras.applications import NASNetLarge,InceptionResNetV2,DenseNet201,NASNetMobile,Xception
from keras import Input,Model
from keras.layers import Dense,Dropout,BatchNormalization,Activation
from keras.regularizers import l2
from utils.dlayers import *
def Inceptresclass(input_shape=(762//2,1024//2,3),classes=21,droprate=0.4,kernel_regu_rate=0.00001):
    inceresnet = InceptionResNetV2(include_top=False,
                              weights='imagenet',
                              input_tensor=None,
                              input_shape=input_shape,
                              pooling='max',
                              classes=1000)
    inceresnet.trainable = False
    x=inceresnet.outputs
    x=Dropout(droprate)(x)
    x=Dense(512, name='dense2048-512')(x)
    x = BatchNormal(x)
    x=Dropout(droprate)(x)
    x=Dense(128, name='dense512-128')(x)
    x=BatchNormal(x)
    x=Dense(classes, activation='softmax', name='out')(x)
    model=Model(inceresnet.inputs,x)
    for l in model.layers:
        l.kernel_regularizer=l2(kernel_regu_rate)
    return model

if __name__=='__main__':
    Inceptresclass()
