import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util

"""
[francesco]
Un biiettore la cui operazione forward ACCETTA UN PARAMETRO `shift`
e restituisce `y = x+shift`.
Necessario in quanto il normale biiettore Shift richiede di
specificare `shift` in fase di costruzione,
ma le nostra policy genererà una media diversa per ciascuna
osservazione, quindi questo è meglio di generare
un nuovo biiettore per ogni singola chiamata alla policy.
"""
class ParametricShift(tfp.bijectors.Bijector):
    def __init__(self, name="parametric_shift"):
        with tf.name_scope(name) as name:    # fa aggiungere un prefisso ai nomi
            super(ParametricShift, self).__init__(
                is_constant_jacobian=True,
                forward_min_event_ndims=0,
                name=name
            )

    @classmethod
    def _is_increasing(cls):
        return True

    def _forward(self, x, shift):
        return x + shift

    def _inverse(self, y, shift):
        return y - shift

    def _forward_log_det_jacobian(self, x, shift):
        return tf.zeros((), dtype=dtype_util.base_dtype(x.dtype))


'''
[francesco]
Un biiettore la cui operazione forward ACCETTA UN PARAMETRO `scale`
e restituisce `y = x*scale`.
Necessario per le stesse ragioni di ParametricShift.
'''
class ParametricScale(tfp.bijectors.Bijector):
    def __init__(self, name="parametric_scale"):
        with tf.name_scope(name) as name:    # fa aggiungere un prefisso ai nomi
            super(ParametricScale, self).__init__(
                is_constant_jacobian=True,
                forward_min_event_ndims=0,
                name=name
            )

    @classmethod
    def _is_increasing(cls):
        return True

    def _forward(self, x, scale):
        return x * scale

    def _inverse(self, y, scale):
        return y / scale

    def _forward_log_det_jacobian(self, x, scale):
        return tf.math.log(tf.abs(scale))
