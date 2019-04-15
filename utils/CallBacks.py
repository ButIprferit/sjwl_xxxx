
from keras.callbacks import  BaseLogger,ReduceLROnPlateau
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping,TensorBoard


lrreduce=ReduceLROnPlateau(monitor='loss',factor=0.2,patience=3,min_lr=0.00001)

class Macro_f1(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
#         val_targ = self.validation_data[1]
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#       print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print(' — val_f1:' ,_val_f1,'macro f1',np.mean(_val_f1))
        return


# TODO
# need finish the callback of print lr

