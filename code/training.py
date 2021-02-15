'''
Modulo che contiene le funzioni di allenamento per f.ni Q, policy, etc.

TODO [francesco]
L'esistenza di questo modulo è provvisoria, l'ho creato solo perché
devo pur mettere la funzione di allenamento policy da qualche parte.
Conviene scegliere la struttura definitiva tra un paio di giorni,
quando avremo un'idea un po' migliore di come si sta sviluppando il progetto.
Alternative ad avere un modulo dedicato al training sono:
 - Mettere le funzioni di training nel modulo principale
   (pro: possiamo accedere alle variabili locali di quel file,
         un vantaggio non trascurabile)
   (contro: il modulo principale potrebbe gonfiarsi)
 - Mettere le funzioni di training come metodi degli oggetti Policy e <funzione q>
   (pro: approccio OOP)
   (contro: non sapremmo dove mettere l'allenamento delle q_targ e (se lo facciamo) di alpha.
            Potrebbero andare nel file principale semmai.)
'''

import tensorflow as tf
import replay_buffer

'''
Il primo passo dell'allenamento delle q,
calcola i valori da usare come bersagli nell'allenamento
La metto da parte per separare questo passo dal resto
RESTITUISCE: i detti valori bersaglio
'''
def compute_q_targets(q1_targ, q2_targ, batch, policy, alpha, gamma):
    # recuperiamo dati dalla batch
    next_observations = batch['next_observations']
    rewards = batch['rewards']
    dones = batch['dones']

    # prediciamo le azioni corrispondenti alle osservazioni future
    next_actions, next_logprobs = policy.compute_actions_and_logprobs(next_observations)
    # calcoliamo i valori della q_targ (come minimo delle due, al solito)
    next_q1_targ_values = q1_targ.compute_q_values(next_observations, next_actions)
    next_q2_targ_values = q2_targ.compute_q_values(next_observations, next_actions)
    next_q_targ_values = tf.reduce_min((next_q1_targ_values, next_q2_targ_values), axis=0)

    # applichiamo l'equazione per il soft value
    next_values = next_q_values - alpha*next_logprobs

    # castiamo per compatibilità con la prossima espressione
    dones = tf.cast(dones, next_values.dtype)

    # equazione per calcolare Q_hat
    targets = rewards + gamma*next_values

    return tf.stop_gradient(targets)

'''
Passo di allenamento di una singola q
RESTITUISCE: valore q e perdita per ciascuna coppai osservazione-azione,
per (eventuali futuri, forse) fini statistici
'''
def trainingstep_single_q(q, batch, target_values, q_optimizer):
    # recupera dati dalla batch
    observations = batch['observations']
    actions = batch['actions']

    # applica una loss MSE standard: calcola le predizioni della NN,
    # calcola il MSE rispetto ai bersagli, poi considera la perdita
    # media e usala per camcolare e applicare i gradienti
    with tf.GradientTape() as tape:
        q_values = q.compute_q_values(observations, actions)
        q_losses = 0.5 * tf.losses.MSE(y_true=target_values, y_pred=q_values)
        q_loss = tf.nn.compute_average_loss(q_losses)
    q_gradients = tape.gradient(q_loss, q.trainable_weights)
    q_optimizer.apply_gradients(zip(
        q_gradients,
        q.trainable_weights
    ))

    return q_values, q_losses

'''
[francesco]
Effettua un passo di allenamento di entrambe le funzioni q
RESTITUISCE: valori calcolati e perdite di entrambe le q (vedi trainingstep_single_q),
per (eventuali futuri, forse) fini statistici
'''
def trainingstep_q(
    q1, q2,
    batch,
    policy,
    q1_targ, q2_targ,
    alpha, gamma,
    q1_optimizer, q2_optimizer
):
    q_targets = compute_q_targets(q1_targ, q2_targ, batch, policy, alpha, gamma)

    q1_values, q1_losses = trainingstep_single_q(q1, batch, target_values, q1_optimizer)
    q2_values, q2_losses = trainingstep_single_q(q2, batch, target_values, q2_optimizer)

    q_values = [q1_values, q2_values]
    q_losses = [q1_losses, q2_losses]
    return q_values, q_losses

'''
[francesco]
Effettua un passo di allenamento della policy:
data una batch di osservazioni, campiona le azioni corrispondenti
e usa la funzione q e le probabilità di quelle azioni
per calcolare la perdita secondo le equazioni del paper,
infine calcola il gradiente della perdita e lo usa per aggiornare i pesi.

RESTITUISCE: la perdita per ciascuna osservazione,
per (eventuali futuri, forse) fini statistici
'''
def trainingstep_policy(policy, batch, q1, q2, alpha, policy_optimizer):
    # Recupera le osservazioni dalla batch
    observations = batch[replay_buffer.OBSERVATIONS]

    # Definiamo la funzione di cui calcolare il gradiente
    # grazie al GradientTape di tf
    with tf.GradientTape() as tape:
        # campioniamo dalla policy azioni e probabilità relative alla batch corrente
        actions, logprobs = policy.compute_actions_and_logprobs(observations)

        # calcola il valore di entrambe le funzioni q...
        q1_target_values = q1.compute_q_values(observations, actions)
        q2_target_values = q2.compute_q_values(observations, actions)
        # e prendi il più piccolo (per ogni osservazione della batch)
        q_target_values = tf.reduce_min((q1_target_values, q2_target_values), axis=0)

        # calcola la perdita di ciascuna coppia osservazione-azione...
        policy_losses = alpha*logprobs - q_target_values
        # ...e la perdita media
        policy_loss = tf.nn.compute_average_loss(policy_losses)

    # Questa funzione controlla che le dimensioni dei tensori usati nel calcolo
    # siano coerenti coi nostri requisiti.
    # In questo caso, controlliamo che tutti i tensori menzionati abbiano
    # lo stesso numero di righe e che logprobs e policy_losses abbiano una
    # sola colonna.
    # Sembra un check di coerenza importante e semplice da fare.
    tf.debugging.assert_shapes((
        (actions, ("R", "A")),
        (logprobs, ("R", 1)),
        (policy_losses, ("R", 1))
    ))

    # Calcola il gradiente della perdita rispetto ai pesi della policy
    policy_gradients = tape.gradient(policy_loss, policy.trainable_weights)

    # Infine applica i gradienti con l'optimizer per aggiornare i pesi della policy
    policy_optimizer.apply_gradients(zip(
        policy_gradients,
        policy.trainable_weights
    ))

    # Dal puro punto di vista dell'allenamento, non c'è bisogno di restituire nulla,
    # restituiamo le perdite della policy per possibili fini statistici
    return policy_losses

'''
Operazione base della prossima funzione
RESTITUISCE: None
'''
def updatestep_single_q_targ(q, q_targ, tau):
    for q_weight, q_targ_weight in zip(q.trainable_weights, q_targ.trainable_weights):
        q_targ_weight.assign(tau*q_weight + (1.0-tau)*q_targ_weight)

'''
[francesco]
Esegue un passo di aggiornamento delle q_targ, tramite media esponenziale mobile
RESTITUISCE: None
'''
def updatestep_q_targ(q1, q2, q1_targ, q2_targ, tau):
    updatestep_single_q_targ(q1, q1_targ, tau)
    updatestep_single_q_targ(q2, q2_targ, tau)
