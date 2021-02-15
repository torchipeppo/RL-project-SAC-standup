# [francesco]

# c'è bisogno di importare un file nella cartella "code",
# quindi devo aggiungerla al path di python
# (l'operazione è locale per questo programma soltanto)
import sys, os
sys.path.append(os.path.abspath('../code'))

import policy as policy_module
import numpy as np
import gym

# Cominciamo creando una piccola policy
# Prima devo definire dei semplici spazi
# fatti senza particolare attenzione
obs_lo = np.full(3, -float('inf'), dtype=np.float32)
obs_hi = np.full(3, float('inf'), dtype=np.float32)
observation_space = gym.spaces.Box(obs_lo, obs_hi)
act_lo = np.full(2, -0.4, dtype=np.float32)
act_hi = np.full(2, 0.4, dtype=np.float32)
action_space = gym.spaces.Box(act_lo, act_hi)

# E ora la policy
polisi = policy_module.Policy(observation_space, action_space)

print(polisi.means_and_sigmas_model.summary())

# Se superiamo questo punto, vuol dire che __init__ è almeno "compilabile".
print()
print("---------- EVVIVA ----------")
print()

# Ora vediamo come si comportano i due/tre metodi che mi preme esporre
# Cominciamo con quello facile
# print(polisi.trainable_weights)
# In realtà è un po' grandicello, mostrami soltanto un elemento
print(polisi.trainable_weights[-2])

# Per provare che succede con la policy deterministica,
# decommentare la prossima riga.
# Memento: dovremo implementare un modo "serio" di cambiare determinismo

# polisi._deterministic = True

# Per il prossimo devo inventarmi qualche osservazione
oss0 = np.array([0.0,0.1,0.2])
oss1 = np.array([1.0,1.1,1.2])
oss2 = np.array([2.0,2.1,2.2])
oss3 = np.array([3.0,3.1,3.2])
oss_batch = np.array([oss0,oss1,oss2,oss3])
# E poi cerchiamo di farci dare azioni e logprob
# Lo facciamo due volte per assicurarci che la policy sia stocastica
actions1, logprobs1 = polisi.compute_actions_and_logprobs(oss_batch)
print("--- [1] ---")
print(actions1)
print(logprobs1)
actions2, logprobs2 = polisi.compute_actions_and_logprobs(oss_batch)
print("--- [2] ---")
print(actions2)
print(logprobs2)

# Infine testiamo il passaggio alla policy deterministica
polisi_det = polisi.create_deterministic_policy()
# Ancora, ripetiamo il test due volte per assicurarci che escano gli stessi risultati
actions3, logprobs3 = polisi_det.compute_actions_and_logprobs(oss_batch)
print("--- [3] ---")
print(actions3)
print(logprobs3)
actions4, logprobs4 = polisi_det.compute_actions_and_logprobs(oss_batch)
print("--- [4] ---")
print(actions4)
print(logprobs4)

# Un'ultima volta con la policy stocastica per controllare che funzioni ancora...
actions5, logprobs5 = polisi.compute_actions_and_logprobs(oss_batch)
print("--- [5] ---")
print(actions5)
print(logprobs5)
# ...e credo di poter essere soddisfatto

print("----- ULTIMA COSA -----")

# controlliamo che i pesi delle due policy siano uguali
print(polisi.trainable_weights[-2])
print(polisi_det.trainable_weights[-2])
print(polisi.trainable_weights[-2] == polisi_det.trainable_weights[-2])
# pare di sì

print("----- NUOVA COSA -----")

# proviamo al volo il nuovo metodo "action"
# lo faccio con la policy deterministica, così posso controllare che
# esca lo stesso risultato
act0 = polisi_det.compute_action(oss0)
print(act0)
# evviva!

# Ovviamente adesso i valori numerici puri non c'azzeccano niente,
# ma voglio solo controllare che i metodi si eseguano senza problemi
print()
print("---------- FINE ----------")
