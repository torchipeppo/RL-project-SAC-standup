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
[francesco]
Effettua un passo di allenamento della policy:
data una batch di osservazioni, campiona le azioni corrispondenti
e usa la funzione q e le probabilità di quelle azioni
per calcolare la perdita secondo le equazioni del paper,
infine calcola il gradiente della perdita e lo usa per aggiornare i pesi.
'''
def trainingstep_policy(policy, batch, q1, q2, alpha, policy_optimizer):
    # TODO [francesco]
    # Ancora non so come conserveremo le funzioni q,
    # provvisoriamente ho presunto due variabili q1 e q2 per semplicità
    # (specialmente per il trainingstep delle q)
    # perché qualcosa devo scrivere,
    # ma possiamo anche decidere di fare una lista/tupla di funzioni q
    # per fare un progetto più generalizzato

    # Recupera le osservazioni dalla batch
    observations = batch[replay_buffer.OBSERVATIONS]

    # Definiamo la funzione di cui calcolare il gradiente
    # grazie al GradientTape di tf
    with tf.GradientTape() as tape:
        # campioniamo dalla policy azioni e probabilità relative alla batch corrente
        actions, logprobs = policy.compute_actions_and_logprobs(observations)

        # calcola il valore di entrambe le funzioni q...
        q1_target_values = __???__  # TODO usare l'API dell'oggetto <funzione q>
        q2_target_values = __???__  # TODO idem
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
    policy_gradiets = tape.gradient(policy_loss, policy.trainable_weights)

    # Infine applica i gradienti con l'optimizer per aggiornare i pesi della policy
    policy_optimizer.apply_gradients(zip(
        policy_gradients,
        policy.trainable_weights
    ))

    # Dal puro punto di vista dell'allenamento, non c'è bisogno di restituire nulla,
    # restituiamo le perdite della policy per possibili fini statistici
    return policy_losses
