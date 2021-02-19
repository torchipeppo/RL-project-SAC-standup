'''
Modulo per una policy gaussiana multivariata

Proprietà/Metodi rilevanti:
 - actions_and_log_probs
 - trainable_variables
 - create_deterministic_policy
'''

import copy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import my_bijectors

class Policy:
    def __init__(
        self,
        observation_space,  #spazio d'osservazione (secondo l'interfaccia di openai gym)
        action_space,       #spazio d'azione (secondo l'interfaccia di openai gym)
        hidden_layer_sizes,             #vedi _make_model
        hidden_acti="relu",             #vedi _make_model
        pseudo_output_acti="linear"     #vedi _make_model
    ):
        # questo parametro decide se le azioni restituite dalla policy debbano
        # essere campionate dalla distribuzione con le medie e varianze
        # date dalla NN, o se invece debba restituire  direttamente le medie,
        # per avere un comportamento più deterministico.
        # questo comportamento deterministico è utile in fase di valutazione
        # per valutare meglio la policy.
        self._deterministic = False

        # questo parametro decide se le azioni
        # (che vengono squashate dalla tanh in [-1,1])
        # debbano essere scalate immediatamente per entrare nello spazio
        # d'azione (che NEL NOSTRO CASO è [-0.4,0.4], ma comunque ce lo
        # faremo passare dall'environment)
        # o se invece debbano essere lasciate in [-1,1] per essere scalate
        # in un secondo momento (magari subito prima di passare l'azione
        # a env.step).
        # questo controllo booleano è stato inserito all'inizio del progetto
        # per poter passare facilmente dall'una all'altra configurazione,
        # adesso serve solo a marcare la decisione presa.
        self._scale_immediately = True

        # estraggo i parametri rilevanti dagli spazi di osservazione e azione
        self._observation_shape = observation_space.shape    # FYI: nel nostro caso è (376,)
        self._action_shape = action_space.shape    # FYI: nel nostro caso, è (17,)
        # ha senso ricordarsi il "vero" action_range solo se abbiamo
        # intenzione di usarlo per scalare le azioni, altrimenti
        # è meglio far finta che il range sia [-1,1]
        # N.B.: Questo modo di settare _action_range e _action_scale
        #       presume che tutte le azioni hanno lo stesso range,
        #       e tale range è simmetrico
        # N.B.: Ho verificato che nel nostro caso tutte e 17 le azioni
        #       hanno lo stesso range simmetrico: [-0.4,0.4]
        if self._scale_immediately:
            self._action_range = (action_space.low.min(), action_space.high.max())
            self._action_scale = self._action_range[1]   # vale a dire "high", un numero positivo
        else:
            # in questo caso questi parametri non verranno usati granché,
            # ma li setto ugualmente per coerenza
            self._action_range = (-1,1)
            self._action_scale = 1

        # la NN della policy.
        # prende in ingresso una batch di osservazioni e restituisce
        # medie e varianze per la distribuzione parametrica da campionare
        # per ottenere le azioni corrispondenti alle osservazioni.
        self.means_and_sigmas_model = _make_model(
            self._observation_shape, hidden_layer_sizes, self._action_shape,
            hidden_acti, pseudo_output_acti
        )

        # la distribuzione usata per campionare azioni non-deterministiche.
        # si tratta di una gaussiana multivariata.
        # la distribuzione di base ha sempre media nulla e (co)varianza unitaria,
        # mentre media e varianza date dalla distribuzione vengono applicate
        # subito dopo con dei biiettori parametrici.
        # non è possibile dare i parametri direttamente alla distribuzione
        # poiché questi devono essere calcolati dalla NN per ogni osservazione,
        # quindi è impossibile assegnare i parametri alla creazione
        # dell'oggetto distribuzione.
        self.base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc = tf.zeros(self._action_shape),    # media
            scale_diag = tf.ones(self._action_shape) # diagonale della matrice diagonale il cui quadrato è la matrice di covarianza
        )

        # definizione del biiettore composto che applica media e deviazione
        # standard al valore campionato dalla distribuzione
        self._apply_means_and_sigmas_bijector = tfp.bijectors.Chain((
            my_bijectors.ParametricShift(name="apply_means"),
            my_bijectors.ParametricScale(name="apply_sigmas"),
        ))
        # applico tale biiettore alla distribuzione base
        self.unclamped_action_distribution = self._apply_means_and_sigmas_bijector(self.base_distribution)

        # definizione del biiettore che fa clamping e forse scaling (vedi _scale_immediately)
        __clamping_bijector = tfp.bijectors.Tanh()     #variabile temporanea, solo per non avere lo stesso biiettore scritto in due punti diversi
        if self._scale_immediately:
            # l'ultimo biiettore specificato è quello eseguito per primo,
            # è lo stesso ordine del prodotto di composizione di funzioni
            self._clamp_and_maybe_scale_actions_bijector = tfp.bijectors.Chain((
                tfp.bijectors.Scale(self._action_scale),   # qui uso lo Scale normale perché voglio sempre portare [-1,1] in [-_action_scale,_action_scale]
                __clamping_bijector,
            ))
        else:
            self._clamp_and_maybe_scale_actions_bijector = __clamping_bijector
        # applico tale biiettore alla distribuzione
        self.action_distribution = self._clamp_and_maybe_scale_actions_bijector(self.unclamped_action_distribution)

    #fine __init__

    '''
    Espone i parametri allenabili della NN (con un paio di alias)
    '''
    @property
    def trainable_weights(self):
        return self.means_and_sigmas_model.trainable_weights
    @property
    def trainable_variables(self):
        return self.trainable_weights
    @property
    def trainable_parameters(self):
        return self.trainable_weights

    '''
    Data una batch di osservazioni, restituisce le azioni corrispondenti
    campionate dalla policy, e il logaritmo della probabilità
    che ciascuna fosse scelta.
    RESTITUISCE: actions, logprobs
    (sì, restituisce due cose)
    '''
    def compute_actions_and_logprobs(self, observations):
        # Recupera la dimensione della batch
        # N.B.: assumo batch monodimensionali (quindi NEL NOSTRO CASO una matrice 2D dove ogni riga è un'osservazione)
        batch_shape = tf.shape(observations)[0:1]
        # prendo solo la prima componente, ma con questa notazione mi assicuro
        # che il tipo di dato rimanga tupla (o array, o tensore, o qualunque esso sia,
        # in ogni caso mi assicuro che non diventi scalare)

        # Invoca il modello per calcolare i parametri della distribuzione
        # per ogni osservazione della batch
        means, sigmas = self.means_and_sigmas_model(observations)

        if self._deterministic:
            # Allora eseguiamo sempre l'azione media, in modo deterministico (per l'appunto).
            # rimane solo clampare (e scalare se così abbiamo deciso)
            actions = self._clamp_and_maybe_scale_actions_bijector(means)
            # Siamo nel caso deterministico: la "probabilità di scegliere quest'azione"
            # è infinita
            # La riga seguente vuol dire: "la dimensione della tabella delle
            # logprob è la stessa di quella delle medie, tranne che l'ultima dimensione
            # (nel nostro caso di matrice 2D, il numero di colonne)
            # è 1 invece dell'originale". Questo perché un'azione è rappresentata
            # da tante componenti (17 nel nostro caso) ma ha associata una sola
            # probabilità.
            logprobs_shape = tf.concat(
                (tf.shape(means)[:-1], [1]),
                axis=0
            )
            # Quest'altra riga invece crea un tensore della dimensione
            # specificata in cui ogni elemento è infinito
            logprobs = tf.fill(logprobs_shape, np.inf)
        else:
            # calcola le azioni campionandole dalla distribuzione multivariata
            # (passando opportuni parametri ai biiettori)
            actions = self.action_distribution.sample(
                batch_shape,
                bijector_kwargs={    # questo parametro serve a passare argomenti ai biiettori
                    "apply_means": {"shift": means},     # al parametro "shift" del biiettore di nome "apply_means" viene passato il vettore means calcolato sopra dalla NN
                    "apply_sigmas": {"scale": sigmas}     # similmente, al parametro "scale" di "apply_sigmas" viene passato sigmas
                }
            )
            # calcola il logaritmo delle probabilità che ciascuna azione
            # sia stata scelta dalla distribuzione (dati anche i parametri dei biiettori)
            logprobs = self.action_distribution.log_prob(
                actions,
                bijector_kwargs={
                    "apply_means": {"shift": means},
                    "apply_sigmas": {"scale": sigmas}
                }
            )[..., tf.newaxis]
            # l'ultima riga aggiunge un nuovo asse come ultimo del tensore.
            #`se il tensore originale era monodimensionale (probabile) lungo N,
            # il nuovo tensore è bidimensionale Nx1
            # (praticamente, un vettore colonna)

        return actions, logprobs

    '''
    Restituisce l'azione corrispondente a una singola osservazione,
    perché il modello vuole necessariamente l'asse della batch.
    RESTITUISCE: l'azione, nel tipo specificato da return_numpy
    '''
    def compute_action(self, observation, return_numpy=True):
        # observation è una riga,
        # dobbiamo trasformarlo in una MATRICE RIGA
        # affinché il modello se lo prenda senza fare storie
        obs_pseudo_batch = observation[np.newaxis, ...]
        # ADESSO possiamo chiamare il modello (col metodo che già abbiamo,
        # così abbiamo pure tutto il postprocessing)...
        act_pseudo_batch, _ = self.compute_actions_and_logprobs(obs_pseudo_batch)
        # e adesso estraiamo l'azione  desiderata.
        # anche act_pseudo_batch è una MATRICE RIGA,
        # mentre io voglio solo la riga
        action = act_pseudo_batch[0]
        if return_numpy:
            return action.numpy()
        else:
            return action

    '''
    Crea una copia profonda della policy, tranne che è deterministica.
    N.B.: Trattandosi di una copia profonda, una volta creata NON si aggiornerà
          automaticamente mentre la policy originale si allena.
          Quindi bisogna creare una nuova copia deterministica
          ogni volta che se ne vuole una, non si può riutilizzare la stessa.
    RESTITUISCE: la copia profonda deterministica
    '''
    def create_deterministic_policy(self):
        twin = copy.deepcopy(self)
        twin.means_and_sigmas_model = keras.models.clone_model(
            self.means_and_sigmas_model
        )
        # pare che clone_model non copi i pesi quindi per sicurezza li clono a mano
        # [ https://stackoverflow.com/questions/54366935/make-a-deep-copy-of-a-keras-model-in-python ]
        twin.means_and_sigmas_model.set_weights(
            self.means_and_sigmas_model.get_weights()
        )
        twin._deterministic = True
        return twin

    '''
    vedi q_function
    '''
    def __getstate__(self):
        state = self.__dict__.copy()
        ms_model = state.pop("means_and_sigmas_model")
        state.update({
            "ms_model_config": ms_model.get_config(),
            "ms_model_weights": ms_model.get_weights(),
        })
        return state

    '''
    vedi q_function
    '''
    def __setstate__(self, state):
        ms_model_config = state.pop("ms_model_config")
        ms_model_weights = state.pop("ms_model_weights")
        ms_model = keras.Model.from_config(ms_model_config)
        ms_model.set_weights(ms_model_weights)
        state["means_and_sigmas_model"] = ms_model
        self.__dict__ = state

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

RESTITUISCE: il modello così costruito
'''
# Ho realizzato questo metodo in modo "statico"
# (nel senso di Java/C#, i.e. senza usare self),
# per cui porto questo metodo fuori dal corpo della classe
def _make_model(observation_shape, hidden_sizes, action_shape, hidden_acti, pseudo_output_acti):

    kl = keras.layers

    # Crea layer di input
    input_layer = kl.Input(shape=observation_shape)
    # Crea tutte le layer nascoste
    hidden_layers = [kl.Dense(size, activation=hidden_acti) for size in hidden_sizes]
    # Crea l'ultima layer densa, che restituisce tutte le medie seguite
    # da tutte le deviazioni standard
    pseudo_output_size = np.prod(action_shape) * 2   #np.prod sta lì solo per rendere il codice un filino più generico,
                                                     #ma dato che il nostro action space sembra essere monodimensionale di fatto non fa nulla.
                                                     #il *2 invece serve perché per ogni azione vogliamo restituire due parametri: media e deviazione standard
    pseudo_output_layer = kl.Dense(pseudo_output_size, activation=pseudo_output_acti)

    # Costruisci la parte "comune" del modello
    common = input_layer
    for hl in hidden_layers:
        common = hl(common)
    common = pseudo_output_layer(common)

    # Separa lo pseudo output a metà: le medie da una parte, le deviazioni dall'altra
    means, sigmas_noact = kl.Lambda(split_a_layer)(common)
    # Applica una attivazione softplus alle deviazioni standard
    sigmas = kl.Lambda(softplus_epsilon)(sigmas_noact)

    # Crea il modello finale
    model = keras.Model(input_layer, (means, sigmas))

    # Restituiscilo
    return model

def split_a_layer(x):
    return tf.split(x, num_or_size_splits=2, axis=-1)

def softplus_epsilon(x):
    return tf.math.softplus(x)+0.00001
