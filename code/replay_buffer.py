'''
[francesco]
Modulo per il replay buffer

I suoi campi sono i seguenti:
 - observations: osservazione vista dall'agente all'inizio dello step,
                 quella su cui si è basato per scegliere l'azione
 - next_observations: osservazione vista dall'agente come conseguenza
                      dell'azione appena eseguita
 - actions: azione selezionata
 - rewards: la reward ottentua a seguito dell'azione
 - dones ("plurale di done"): segnale booleano di terminazione ricevuto dopo l'azione
'''

import numpy as np

# le chiavi usate nei dizionari restituiti dai ReplayBuffer
OBSERVATIONS = "observations"
NEXT_OBSERVATIONS = "next_observations"
ACTIONS = "actions"
REWARDS = "rewards"
DONES = "dones"

class ReplayBuffer:
    '''
    Crea un nuovo buffer.
    N.B.: observation_size e action_size devono essere scalari.
          Fortunatamente, nel nostro caso lo sono.
    '''
    def __init__(self, observation_size, action_size, buffer_size):
        # buffer per i cinque campi, vedi sopra
        self.observations = np.zeros([buffer_size, observation_size])
        self.next_observations = np.zeros([buffer_size, observation_size])
        self.actions = np.zeros([buffer_size, action_size])
        self.rewards = np.zeros(buffer_size)
        self.dones = np.zeros(buffer_size)
        # indice della prossima cella da riempire
        self.write_idx = 0
        # numero di elementi attualmente nel buffer
        self.size = 0
        # massima capacità del buffer:
        # una volta superata, elimineremo gli elementi più vecchi
        self.max_size = buffer_size

    '''
    Aggiunge una nuova entry al buffer.
    Se è pieno, l'entry più vecchia viene sovrascritta.
    '''
    def store(self, obs, next_obs, act, rew, done):
        # scrivi nel buffer alla cella indicata
        self.observations[self.write_idx] = obs
        self.next_observations[self.write_idx] = next_obs
        self.actions[self.write_idx] = act
        self.rewards[self.write_idx] = rew
        self.dones[self.write_idx] = done
        # fai avanzare l'indice di scrittura.
        # una volta raggiunta la fine, riparti dall'inizio:
        # questo fa in modo che sovrascriviamo le entry nello stesso ordine
        # in cui le abbiamo scritte, quindi l'entry cancellata sarà sempre
        # la più vecchia
        self.write_idx = (self.write_idx+1) % self.max_size
        # incrementa il numero di elementi nel buffer,
        # a meno che la condizione non sia falsa,
        # nel qual caso il buffer era già pieno
        # e abbiamo sovrascritto una vecchia entry
        if self.size < self.max_size:
            self.size += 1

    '''
    Estrae una batch casuale dal buffer.
    `batch_size` è il numero di elementi da estrarre.
    '''
    def random_batch(self, batch_size):
        # estrai `batch_size` indici casuali
        randomized_indices = np.random.randint(0, self.size, batch_size)
        # recupera i dati corrispondenti a quegli indici dal buffer
        # (grazie all'indicizzamento multiplo di numpy)
        obs_batch = self.observations[randomized_indices]
        next_obs_batch = self.next_observations[randomized_indices]
        act_batch = self.actions[randomized_indices]
        rew_batch = self.rewards[randomized_indices]
        done_batch = self.dones[randomized_indices]
        # restituisci l'intera batch sotto forma di dizionario
        return {
            OBSERVATIONS: obs_batch,
            NEXT_OBSERVATIONS: next_obs_batch,
            ACTIONS: act_batch,
            REWARDS: rew_batch,
            DONES: done_batch
        }
