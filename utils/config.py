# this script is define the parmeta

from models.Xception_Classification import XceptionClass
from models.NASLARGE_Classification import NASLARGEClass
from models.InceptResnetClassification import InceptresClass

num_gpu=1

droprate=0.4

kernel_re=0.000001

nb_classes = 14

modelname='Xception'

modeldict={'Xception':XceptionClass,
           'NASLARGEClass':NASLARGEClass,
           'InceptResnetClass':InceptresClass
           }

lr=0.01

img_h,img_w=762//2,1024//2

batch_size=16

nb_epochs=50

dataset_dir='/Disk4/xkp/dataset/iwilddata'

root_dir='/Disk4/xkp/project/sjwl_xxxx'

train_images_dir=dataset_dir+'/train_images'

train_csv_path=dataset_dir+'/train.csv'

test_images_dir=dataset_dir+'/test_images'

test_csv_path=dataset_dir+'/test.csv'

weightsname=None