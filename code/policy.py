'''
[francesco]
Modulo per una policy gaussiana multivariata

TODO
Devo esporre le seguenti proprietà (non necessariamente con questi nomi):
 - actions_and_log_probs
 - trainable_variables
'''

import tensorflow as tf
import tensorflow.keras as keras

kl = keras.layers

class Policy:
    def __init__(self, ???):
        pass

    '''
    Crea una rete neurale che prende in ingresso una o più osservazioni
    e restituisce i parametri corrispondenti della gaussiana.
    La faccio in forma parametrica per renderla più facile da modificare.

    Parametri:
    observation_shape : Tuple
        Le dimensioni di una singola osservazione in input.
        Una volta creata la rete, per darle più osservazioni
        basterà dare un tensore con una dimensione in più.
        NEL NOSTRO CASO, le osservazioni sono array di numpy monodimensionali
        (o tensori di tensorflow monodimensionali),
        quindi una batch è rappresentata da un array/tensore bidimensionale.
    hidden_sizes : Sequence of Ints
        Una sequenza di interi, ciascuna delle quali rappresenta le dimensioni
        di una layer nascosta. Il numero di layer nascoste sarà quindi
        pari al numero di elementi nella sequenza. Le dimensioni vanno date in
        ordine bottom-to-top (dalla layer più vicina all'input a quella più vicina
        all'output).
    action_shape : Tuple
        La dimensione dello spazio d'azione.
        NEL NOSTRO CASO, è anch'esso monodimensionale.
    hidden_act : String or Function
    pseudo_output_act: String or Function
        Funzioni di attivazione delle layer nascoste e dell'ultima layer densa,
        rispettivamente.
    '''
    def _make_model(observation_shape, hidden_sizes, action_shape, hidden_act, pseudo_output_act):
        # TODO potremmo aver bisogno del cast_and_concat perché mi pare di capire che le osservazioni siano float64 mentre le azioni float32

        # Crea layer di input
        input_layer = layers.Input(shape=observation_shape)
        # Crea tutte le layer nascoste
        hidden_layers = [layers.Dense(size, activation=hidden_act) for size in hidden_sizes]
        # Crea l'ultima layer densa, che restituisce tutte le medie seguite
        # da tutte le deviazioni standard
        pseudo_output_size = np.prod(action_shape) * 2   #np.prod sta lì solo per rendere il codice un filino più generico,
                                                         #ma dato che il nostro action space sembra essere monodimensionale di fatto non fa nulla.
                                                         #il *2 invece serve perché per ogni azione vogliamo restituire due parametri: media e deviazione standard
        pseudo_output_layer = layers.Dense(pseudo_output_size, activation=pseudo_output_act)

        # Costruisci la parte "comune" del modello
        common = input_layer
        for hl in hidden_layers:
            common = hl(common)
        common = pseudo_output_layer(common)

        # Separa lo pseudo output a metà: le medie da una parte, le deviazioni dall'altra
        split_a_layer = lambda x: tf.split(x, num_or_size_of_splits=2, axis=-1)
        means, sigmas_noact = layers.Lambda(split_a_layer)(common)
        # Applica una attivazione softplus alle deviazioni standard
        softplus_epsilon = lambda x: tf.math.softplus(x)+0.00001
        sigmas = layers.Lambda(softplus_epsilon)(sigmas_noact)

        # Crea il modello finale
        model = keras.Model(input_layer, (means, sigmas))

        # Restituiscilo
        return model
