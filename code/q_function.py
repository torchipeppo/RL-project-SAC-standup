'''
Modulo per una funzione q

Proprietà/Metodi rilevanti:
 - compute_q_values
 - trainable_variables
 - create_deepcopy
'''

import tensorflow as tf
import tensorflow.keras as keras
import copy

class Q_Function:
    def __init__(
        self,
        observation_space,  #spazio d'osservazione (secondo l'interfaccia di openai gym)
        action_space,       #spazio d'azione (secondo l'interfaccia di openai gym)
        hidden_layer_sizes,      #vedi _make_model
        hidden_acti="relu",      #vedi _make_model
        output_acti="linear"     #vedi _make_model
    ):
        # estraggo i parametri rilevanti dagli spazi di osservazione e azione
        # (stavolta ci servono solo le shape, lascio comunque tutto lo space come
        #  argomento per "coerenza" con l'API della Policy)
        self._observation_shape = observation_space.shape    # FYI: nel nostro caso è (376,)
        self._action_shape = action_space.shape    # FYI: nel nostro caso, è (17,)

        # crea il modello
        self.q_model = _make_model(
            self._observation_shape, self._action_shape,
            hidden_layer_sizes, hidden_acti, output_acti
        )

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
    Data una batch di osservazioni e di azioni,
    RESTITUISCE i valori q corrispondenti a ciascuna coppia osservazione-azione.
    '''
    def compute_q_values(self, observations, actions):
        vals = self.q_model((observations, actions))
        return vals

    '''
    Crea una copia di quest'oggetto che non ha legami con l'originale.
    Utile per inizializzare la q_targ corrispondente alla q
    RESTITUISCE: la copia
    '''
    def create_deepcopy(self):
        twin = copy.deepcopy(self)
        twin.q_model = keras.models.clone_model(self.q_model)
        # pare che clone_model non copi i pesi quindi per sicurezza li clono a mano
        # [ https://stackoverflow.com/questions/54366935/make-a-deep-copy-of-a-keras-model-in-python ]
        twin.q_model.set_weights(self.q_model.get_weights())
        return twin

    '''
    decide cosa salvare in fase di pickling
    il fatto è che non possiamo picklare il modello,
    e se anche si potesse probabilmente uscirebbe male
    '''
    def __getstate__(self):
        # lo stato da salvare è praticamente quello di default...
        state = self.__dict__.copy()
        # eccetto che bisogna escludere il modello...
        q_model = state.pop("q_model")
        # e al suo posto ci aggiungiamo le informazioni utili a ricrearlo successivamente
        state.update({
            "q_model_config": q_model.get_config(),
            "q_model_weights": q_model.get_weights(),
        })
        return state

    '''
    dice come ricostruire l'oggetto in fase di unpickling
    '''
    def __setstate__(self, state):
        # recupera config e pesi ed eliminali dal dizionario di stato
        q_model_config = state.pop("q_model_config")
        q_model_weights = state.pop("q_model_weights")
        # ricostruisci il modello sulla base di quelle informazioni
        q_model = keras.Model.from_config(q_model_config)
        q_model.set_weights(q_model_weights)
        # inserisci il modello nel dizionario di stato...
        state["q_model"] = q_model
        # ...e assegna il dizionario di stato al nuovo oggetto,
        # completando il caricamento
        self.__dict__ = state

'''
Crea una rete neurale che prende in ingresso una (batch di) coppia
osservazione azione e restutisce il (batch di) valore Q corrispondente

I parametri solo tali e quali a quelli della policy
(ma l'ordine è leggermente diverso!)

RESTITUISCE: la rete neurale
'''
def _make_model(observation_shape, action_shape, hidden_sizes, hidden_acti, output_acti):

    kl = keras.layers

    # Crea layer di input
    input_layers = [kl.Input(shape=observation_shape), kl.Input(shape=action_shape)]
    concatenate_layer = kl.Concatenate()

    # Crea tutte le layer nascoste
    hidden_layers = [kl.Dense(size, activation=hidden_acti) for size in hidden_sizes]

    # Layer di output
    output_layer = kl.Dense(1, activation=output_acti)

    # Costruiamo tutto il modello
    out = input_layers
    out = concatenate_layer(out)
    for hl in hidden_layers:
        out = hl(out)
    out = output_layer(out)

    model = keras.Model(input_layers, out)

    return model
