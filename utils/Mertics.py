# coding: utf8
from keras.callbacks import Callback
from sklearn.metrics import f1_score
import numpy as np
import tensorflow as tf
def get_lr_metric(optimizer):
    '''
    return the learn rate of the train process
    :param optimizer:
    :return:
    '''
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

from keras import backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def MacroF1(y_true,y_pred):
    val_predict = np.argmax(np.asarray(y_pred),)
    print(np.asarray(y_true).shape)
    #         val_targ = self.validation_data[1]
    val_targ = np.argmax(np.asarray(y_true),axis=0)
    _val_f1 = f1_score(val_targ, val_predict, average='macro')
    #         _val_recall = recall_score(val_targ, val_predict)
    #         _val_precision = precision_score(val_targ, val_predict)

    #         self.val_recalls.append(_val_recall)
    #         self.val_precisions.append(_val_precision)
    #       print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
  #  print(' — val_f1:', _val_f1)
    return _val_f1

def MAP(classes):
    '''

    :param y_true:
    :param y_pre:
    :return: the mean average precision value
    '''
    def map_funct(y_true,y_pre):
        map,_=tf.metrics.sparse_average_precision_at_k(y_true,y_pre,k=classes)
        return map
    return map_funct

def calculate_mAP(y_true,y_pred):
    num_classes = y_true.shape[1]
    average_precisions = []
    relevant = K.sum(K.round(K.clip(y_true, 0, 1)))
    tp_whole = K.round(K.clip(y_true * y_pred, 0, 1))
    for index in range(num_classes):
        temp = K.sum(tp_whole[:,:index+1],axis=1)
        average_precisions.append(temp * (1/(index + 1)))
    AP = Add()(average_precisions) / relevant
    mAP = K.mean(AP,axis=0)
    return mAP