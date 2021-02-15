'''
Modulo per una funzione q

Proprietà/Metodi rilevanti:
 - values
 - trainable_variables
'''

import tensorflow as tf
import tensorflow.keras as keras

class Q_Function:
    def __init__(
        self,
        observation_space,  #spazio d'osservazione (secondo l'interfaccia di openai gym)
        action_space,       #spazio d'azione (secondo l'interfaccia di openai gym)
        hidden_layer_sizes=(256,256),   #vedi _make_model
        hidden_acti="relu",             #vedi _make_model
        pseudo_output_acti="linear"     #vedi _make_model
    ):
        # estraggo i parametri rilevanti dagli spazi di osservazione e azione
        self._observation_shape = observation_space.shape;    # FYI: nel nostro caso è (376,)
        self._action_shape = action_space.shape;    # FYI: nel nostro caso, è (17,)

        # crea il modello
        self.q_model = _make_model(observation_shape, action_shape, hidden_layer_sizes, hidden_acti, output_acti)

    '''
    Espone i parametri allenabili della NN (con un paio di alias)
    '''
    @property
    def trainable_weights(self):
        return self.q_model.trainable_weights
    @property
    def trainable_variables(self):
        return self.trainable_weights
    @property
    def trainable_parameters(self):
        return self.trainable_weights

    '''
    Data una batch di osservazioni e di azioni, RESTITUISCE i valori q
    corrispondenti a ciascuna coppia osservazione-azione.
    '''
    def values(self, observations, actions):
        vals = self.q_model((observations, actions))
        return vals

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
