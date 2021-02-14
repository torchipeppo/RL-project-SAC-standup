'''
Modulo per una funzione q

Proprietà/Metodi rilevanti:
 - values [TODO]
 - trainable_variables [TODO]
'''

import tensorflow as tf
import tensorflow.keras as keras

class Q_Function:
    def __init__(self, ...):
        pass

'''
Crea una rete neurale che prende in ingresso una (batch di) coppia
osservazione azione e restutisce il (batch di) valore Q corrispondente

I parametri solo tali e quali a quelli della policy

RESTITUISCE: la rete neurale
'''
def _make_model(observation_shape, action_shape, hidden_sizes, hidden_acti, output_acti):

    kl = keras.layers

    # Crea layer di input
    imput_layers = (kl.Input(shape=observation_shape), kl.Input(shape=action_shape))

    # TODO potremmo aver bisogno del cast_and_concat perché mi pare di capire che le osservazioni siano float64 mentre le azioni float32
    # (vedi: https://github.com/rail-berkeley/softlearning/blob/master/softlearning/utils/tensorflow.py#L32)

    # Crea tutte le layer nascoste
    hidden_layers = [kl.Dense(size, activation=hidden_acti) for size in hidden_sizes]

    # Layer di output
    output_layer = kl.Dense(1, activation=output_acti)

    # Costruiamo tutto il modello
    out = input_layers
    for hl in hidden_layers:
        out = hl(out)
    out = output_layer(out)

    model = keras.Model(input_layers, out)

    return model
