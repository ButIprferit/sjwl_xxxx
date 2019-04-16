from models.NASLARGE_Classification import *
import os 
os.environ["CUDA_VISIBLE_DEVICES"]='1'
model=NASNetClass(input_shape=(762//2,1024//2,3),include_top=False,weights='imagenet')
# for l in model.layers:
#     print l.name
for i in range(1039):
    model.layers[i].trainable=False
for i in range(len(model.layers)):
    print(i,model.layers[i].name,model.layers[i].trainable)

# model.summary()
# from keras.utils import plot_model
# plot_model(model, to_file='model.png')
# # from keras.models import Model
# # x=Model()
# # x.load_weights('dasd',by_name=True,skip_mismatch=True)
model.save_weights('model.h5')