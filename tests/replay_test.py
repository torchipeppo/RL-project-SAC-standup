# c'è bisogno di importare un file nella cartella "code",
# quindi devo aggiungerla al path di python
# (l'operazione è locale per questo programma soltanto)
import sys, os
sys.path.append(os.path.abspath('../code'))

import replay_buffer
import numpy as np

# costanti
OBS_SIZE=3
ACT_SIZE=2
BUF_SIZE=5

# helper per generare dati
def make_obs(num):
    return np.array([10*num+i for i in (0,1,2)])
def make_act(num):
    return np.array([100*num+i for i in (0,1)])

# Creiamo un piccolo buffer
rb = replay_buffer.ReplayBuffer(OBS_SIZE, ACT_SIZE, BUF_SIZE)

# Riempiamolo parzialmente
rb.store(
    make_obs(1),
    make_obs(2),
    make_act(1),
    1.12,
    False
)
rb.store(
    make_obs(2),
    make_obs(3),
    make_act(2),
    2.67,
    False
)
rb.store(
    make_obs(3),
    make_obs(4),
    make_act(3),
    3.99999,
    True
)

# Stampiamo un po' di parametri
print("---------- PARZIALE ----------")
print("size: {}".format(rb.size))
print("rewards: {}".format(rb.rewards))
print("a random batch: {}".format(rb.random_batch(2)))

# Riempiamo il buffer fino a sovrascrivere
rb.store(
    make_obs(4),
    make_obs(5),
    make_act(4),
    4.71,
    False
)
rb.store(
    make_obs(5),
    make_obs(6),
    make_act(5),
    5.86,
    False
)
rb.store(
    make_obs(6),
    make_obs(7),
    make_act(6),
    6.00,
    False
)
rb.store(
    make_obs(7),
    make_obs(8),
    make_act(7),
    7.7902,
    True
)

# Stampiamo le stesse statistiche
print("---------- TOTALE ----------")
print("size: {}".format(rb.size))
print("rewards: {}".format(rb.rewards))
print("a random batch: {}".format(rb.random_batch(3)))
