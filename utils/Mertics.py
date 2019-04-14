# coding: utf8
def get_lr_metric(optimizer):
    '''
    return the learn rate of the train process
    :param optimizer:
    :return:
    '''
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr