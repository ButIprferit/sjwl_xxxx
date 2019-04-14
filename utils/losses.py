# coding: utf8
from keras.losses import categorical_crossentropy,sparse_categorical_crossentropy
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import _to_tensor
from keras.backend.common import epsilon
def weights_crossentropy(y_true,y_pred,weights):
    # need cacluate the ratio of different class in the dataset
    # weights=np.ones(shape=(1,14))
    # weights[0,0]=0.125
    # weights=tf.Variable(weights,dtype=y_pred.dtype.base_dtype)
    def categorical_crossentropy(target, output, from_logits=False, axis=-1,weights=weights):
        """Categorical crossentropy between an output tensor and a target tensor.

        # Arguments
            target: A tensor of the same shape as `output`.
            output: A tensor resulting from a softmax
                (unless `from_logits` is True, in which
                case `output` is expected to be the logits).
            from_logits: Boolean, whether `output` is the
                result of a softmax, or is a tensor of logits.
            axis: Int specifying the channels axis. `axis=-1`
                corresponds to data format `channels_last`,
                and `axis=1` corresponds to data format
                `channels_first`.

        # Returns
            Output tensor.

        # Raises
            ValueError: if `axis` is neither -1 nor one of
                the axes of `output`.
        """
        output_dimensions = list(range(len(output.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(output.get_shape()))))
        # Note: tf.nn.softmax_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # scale preds so that the class probas of each sample sum to 1
            # manual computation of crossentropy
            _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            return - tf.reduce_sum(target * tf.log(output)*weights, axis)
def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_sum(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed




# TODO
# In the imati competation can output 21 value and 1000+ value embedding to 21 ,then cacl the embedding loss.
