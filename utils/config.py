# this script is define the parmeta

from models.Xception_Classification import XceptionClass
from models.NASLARGE_Classification import NASLARGEClass
from models.InceptResnetClassification import InceptresClass
class config_par(object):
    def __init__(self):


        self.num_gpu=1

        self.droprate=0.4

        self.kernel_re=0.000001

        self.nb_classes = 14

        self.modelname='Xception'

        self.modeldict={'Xception':XceptionClass,
                   'NASLARGEClass':NASLARGEClass,
                   'InceptResnetClass':InceptresClass
                   }

        self.lr=0.01

        self.img_h=762//2
        self.img_w=1024//2

        self.batch_size=8

        self.nb_epochs=50

        self.dataset_dir='/Disk4/xkp/dataset/iwilddata'

        self.root_dir='/Disk4/xkp/project/sjwl_xxxx'

        self.train_images_dir=self.dataset_dir+'/train_images'

        self.train_csv_path=self.dataset_dir+'/train.csv'

        self.test_images_dir=self.dataset_dir+'/test_images'

        self.test_csv_path=self.dataset_dir+'/test.csv'

        self.weightsname=None

config=config_par()