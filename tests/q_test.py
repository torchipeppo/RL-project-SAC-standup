# [francesco]

# c'è bisogno di importare un file nella cartella "code",
# quindi devo aggiungerla al path di python
# (l'operazione è locale per questo programma soltanto)
import sys, os
sys.path.append(os.path.abspath('../code'))

import q_function as q_fn_module
import numpy as np
import gym

# Cominciamo creando una piccola q
# Prima devo definire dei semplici spazi
# fatti senza particolare attenzione
obs_lo = np.full(3, -float('inf'), dtype=np.float32)
obs_hi = np.full(3, float('inf'), dtype=np.float32)
observation_space = gym.spaces.Box(obs_lo, obs_hi)
act_lo = np.full(2, -0.4, dtype=np.float32)
act_hi = np.full(2, 0.4, dtype=np.float32)
action_space = gym.spaces.Box(act_lo, act_hi)

# E ora la policy
q = q_fn_module.Q_Function(observation_space, action_space)

print(q.q_model.summary())

# Se superiamo questo punto, vuol dire che __init__ è almeno "compilabile".
print()
print("---------- EVVIVA ----------")
print()

# Ora vediamo come si comportano i due/tre metodi che mi preme esporre
# Cominciamo con quello facile
# print(polisi.trainable_weights)
# In realtà è un po' grandicello, mostrami soltanto un elemento
print(q.trainable_weights[-1])

# Per il prossimo devo inventarmi qualche osservazione e azione
oss0 = np.array([0.0,0.1,0.2])
oss1 = np.array([1.0,1.1,1.2])
oss2 = np.array([2.0,2.1,2.2])
oss3 = np.array([3.0,3.1,3.2])
oss_batch = np.array([oss0,oss1,oss2,oss3])
azn0 = np.array([0.0,0.1])*0.01
azn1 = np.array([1.0,1.1])*0.01
azn2 = np.array([2.0,2.1])*0.01
azn3 = np.array([3.0,3.1])*0.01
azn_batch = np.array([azn0,azn1,azn2,azn3])
# E poi cerchiamo di farci dare i valori q
values = q.compute_q_values(oss_batch, azn_batch)
print("--- [*] ---")
print(values)


# Ovviamente adesso i valori numerici puri non c'azzeccano niente,
# ma voglio solo controllare che i metodi si eseguano senza problemi
print()
print("---------- FINE ----------")
