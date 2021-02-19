'''
Questo modulo contiene gli oggetti per le q e la policy
e tutte le funzioni che le riguardano, come ad esempio il training.

Questa classe non è pensata come un modulo stagno,
SAC potrà accedere a tutti i suoi membri in caso di necessità
(ma lo farà sempre soltanto in lettura)
'''

import numpy as np
import tensorflow as tf
import pickle
import q_function as q_fn_module
import policy as policy_module
import replay_buffer as repbuf_module # per le costanti

# costanti per il dizionario restituito da trainingstep
Q1_LOSS = "q1_loss"
Q2_LOSS = "q2_loss"
POLICY_LOSS = "policy_loss"

class Agent:
    def __init__(
        self,
        # spazi di osservazione e azione
        observation_space, action_space,
        # vedi sac.py
        hidden_layer_sizes,
        q_lr, policy_lr, alpha, gamma, tau
    ):
        # creiamo le q, la policy e le q_targ
        self.q1 = q_fn_module.Q_Function(
            observation_space, action_space,
            hidden_layer_sizes
        )
        self.q2 = q_fn_module.Q_Function(
            observation_space, action_space,
            hidden_layer_sizes
        )
        self.policy = policy_module.Policy(
            observation_space, action_space,
            hidden_layer_sizes
        )
        self.q1_targ = self.q1.create_deepcopy()
        self.q2_targ = self.q2.create_deepcopy()
        # inizializzazione optimizer
        self.q1_optimizer = tf.optimizers.Adam(learning_rate=q_lr)
        self.q2_optimizer = tf.optimizers.Adam(learning_rate=q_lr)
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=policy_lr)
        # altri parametri
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

    #############################################################
    #####################   TRAINING    #########################
    #############################################################

    '''
    Il primo passo dell'allenamento delle q,
    calcola i valori da usare come bersagli nell'allenamento
    (la Q_hat del paper)
    La metto da parte per separare questo passo dal resto
    RESTITUISCE: i detti valori bersaglio
    '''
    def compute_q_targets(self, batch):
        # recuperiamo dati dalla batch
        next_observations = batch[repbuf_module.NEXT_OBSERVATIONS]
        rewards = batch[repbuf_module.REWARDS]
        dones = batch[repbuf_module.DONES]

        # prediciamo le azioni corrispondenti alle osservazioni future
        next_actions, next_logprobs = self.policy.compute_actions_and_logprobs(next_observations)

        # calcoliamo i valori della q_targ per le coppie osservazione-azione
        # "future" così ottenute (prendendo il minimo delle due, al solito)
        next_q1_targ_values = self.q1_targ.compute_q_values(next_observations, next_actions)
        next_q2_targ_values = self.q2_targ.compute_q_values(next_observations, next_actions)
        next_q_targ_values = tf.reduce_min((next_q1_targ_values, next_q2_targ_values), axis=0)

        # applichiamo l'equazione per il soft value
        next_v_values = next_q_targ_values - self.alpha*next_logprobs

        # cast per compatibilità con la prossima espressione
        dones = tf.cast(dones, next_v_values.dtype)

        # equazione per calcolare Q_hat
        targets = rewards + self.gamma*(1-dones)*next_v_values

        return tf.stop_gradient(targets)

    '''
    Passo di allenamento di una singola q
    RESTITUISCE: la perdita media,
    per (eventuali futuri, forse) fini statistici
    '''
    # Nota: questa funzione prende argomenti anziché usare i valori salvati in self
    # perché viene chiamata una volta per q1 e una volta per q2
    def trainingstep_single_q(self, q, batch, target_values, q_optimizer):
        # recupera dati dalla batch
        observations = batch[repbuf_module.OBSERVATIONS]
        actions = batch[repbuf_module.ACTIONS]

        # applica una loss MSE standard: calcola le predizioni della NN,
        # calcola il MSE rispetto ai bersagli, poi considera la perdita
        # media e usala per calcolare e applicare i gradienti
        with tf.GradientTape() as tape:
            q_values = q.compute_q_values(observations, actions)
            q_losses = 0.5 * tf.losses.MSE(y_true=target_values, y_pred=q_values)
            q_loss = tf.nn.compute_average_loss(q_losses)
        q_gradients = tape.gradient(q_loss, q.trainable_weights)
        q_optimizer.apply_gradients(zip(
            q_gradients,
            q.trainable_weights
        ))

        # return q_values, q_losses
        return q_loss

    '''
    Effettua un passo di allenamento di entrambe le funzioni q
    RESTITUISCE: valori calcolati e perdite di entrambe le q (vedi trainingstep_single_q),
    per (eventuali futuri, forse) fini statistici
    '''
    def trainingstep_q(self, batch):
        target_values = self.compute_q_targets(batch)

        q1_loss = self.trainingstep_single_q(self.q1, batch, target_values, self.q1_optimizer)
        q2_loss = self.trainingstep_single_q(self.q2, batch, target_values, self.q2_optimizer)

        q_loss = [q1_loss, q2_loss]
        return q_loss

    '''
    Effettua un passo di allenamento della policy:
    data una batch di osservazioni, campiona le azioni corrispondenti
    e usa la funzione q e le probabilità di quelle azioni
    per calcolare la perdita secondo le equazioni per J_pi del paper,
    infine calcola il gradiente della perdita e lo usa per aggiornare i pesi.

    RESTITUISCE: la perdita media,
    per (eventuali futuri, forse) fini statistici
    '''
    def trainingstep_policy(self, batch):
        # Recupera le osservazioni dalla batch
        observations = batch[repbuf_module.OBSERVATIONS]

        # Definiamo la funzione di cui calcolare il gradiente
        # grazie al GradientTape di tf
        with tf.GradientTape() as tape:
            # campioniamo dalla policy (stocastica)
            # azioni e probabilità relative alla batch corrente
            actions, logprobs = self.policy.compute_actions_and_logprobs(observations)

            # calcola il valore q di ogni coppia osservazione-azione
            # come minimo delle due stime
            q1_target_values = self.q1.compute_q_values(observations, actions)
            q2_target_values = self.q2.compute_q_values(observations, actions)
            q_target_values = tf.reduce_min((q1_target_values, q2_target_values), axis=0)

            # calcola la perdita di ciascuna coppia osservazione-azione
            # applicando l'equazione per J_pi...
            policy_losses = self.alpha*logprobs - q_target_values
            # ...e poi calcola la perdita media
            policy_loss = tf.nn.compute_average_loss(policy_losses)

        # Calcola il gradiente della perdita rispetto ai pesi della policy
        policy_gradients = tape.gradient(policy_loss, self.policy.trainable_weights)

        # Infine applica i gradienti con l'optimizer per aggiornare i pesi della policy
        self.policy_optimizer.apply_gradients(zip(
            policy_gradients,
            self.policy.trainable_weights
        ))

        # Dal puro punto di vista dell'allenamento, non c'è bisogno di restituire nulla,
        # restituiamo la perdita della policy per possibili fini statistici
        return policy_loss#es

    '''
    Operazione base della prossima funzione
    RESTITUISCE: None
    '''
    # Ancora, questa prende un po' di parametri anziché usare quelli in self
    # perché è chiamata per q1 e q2
    def updatestep_single_q_targ(self, q, q_targ):
        for q_weight, q_targ_weight in zip(q.trainable_weights, q_targ.trainable_weights):
            q_targ_weight.assign(self.tau*q_weight + (1.0-self.tau)*q_targ_weight)

    '''
    Esegue un passo di aggiornamento delle q_targ, tramite media esponenziale mobile
    RESTITUISCE: None
    '''
    def updatestep_q_targ(self):
        self.updatestep_single_q_targ(self.q1, self.q1_targ)
        self.updatestep_single_q_targ(self.q2, self.q2_targ)

    '''
    esegue un singolo passo di training,
    aggiornando tutte le NN
    vedi le funzioni sopra per i dettagli
    RESTITUISCE: un dizionario con le statistiche restituite
    dai singoli trainingstep, per (eventuali) fini statistici
    '''
    def trainingstep(self, batch):
        q_loss = self.trainingstep_q(batch)
        policy_loss = self.trainingstep_policy(batch)
        self.updatestep_q_targ()
        # per l'allenamento in sé non abbiamo bisogno di restituire nulla,
        # per (possibili) fini statistici restituiamo le perdite delle NN
        return {
            Q1_LOSS: q_loss[0],
            Q2_LOSS: q_loss[1],
            POLICY_LOSS: policy_loss,
        }

    #############################################################
    #################   AUSILIARIE VARIE    #####################
    #############################################################

    '''
    espone compute_action, vedi Policy
    '''
    def compute_action(self, obs):
        return self.policy.compute_action(obs)

    '''
    salva tutti i modelli
    '''
    def save_all_models(self, save_path, suffix):
        if not save_path.exists():
            save_path.mkdir()
        with open(save_path/"q1{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.q1, f)
        with open(save_path/"q2{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.q2, f)
        with open(save_path/"policy{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.policy, f)
        with open(save_path/"q1targ{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.q1_targ, f)
        with open(save_path/"q2targ{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.q2_targ, f)
        # ora non ho più bisogno di salvare i modelli separatamente
        # perché picklo pesi e configurazioni in __getstate__
        # e ripristino tutto in __setstate__
