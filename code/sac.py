import numpy as np
import tensorflow as tf
import gym
import time
import datetime
from pathlib import Path
import policy as policy_module
import q_function as q_fn_module
import replay_buffer as replay_buffer_module
import training as training_module
import path_constants

'''
Funzioni ausiliarie
'''

def save_everything(save_path, suffix):
    import pickle
    base_path = save_path
    with open(base_path/"q1{}.pkl".format(suffix), "w") as f:
        pickle.dump(q1, f)
    with open(base_path/"q2{}.pkl".format(suffix), "w") as f:
        pickle.dump(q2, f)
    with open(base_path/"policy{}.pkl".format(suffix), "w") as f:
        pickle.dump(policy, f)
    with open(base_path/"q1targ{}.pkl".format(suffix), "w") as f:
        pickle.dump(q1targ, f)
    with open(base_path/"q2targ{}.pkl".format(suffix), "w") as f:
        pickle.dump(q2targ, f)
    with open(base_path/"replaybuffer{}.pkl".format(suffix), "w") as f:
        pickle.dump(replay_buffer, f)
    # poi ho l'impressione che dovrei salvare i modelli separatamente, per sicurezza
    policy.means_and_sigmas_model.save(base_path/"policy{}.h5".format(suffix))
    q1.q_model.save(base_path/"q1{}.h5".format(suffix))
    q2.q_model.save(base_path/"q2{}.h5".format(suffix))
    q1targ.q_model.save(base_path/"q1targ{}.h5".format(suffix))
    q2targ.q_model.save(base_path/"q2targ{}.h5".format(suffix))

def simple_episode_info_dump(logfpath, episode_length, episode_return):
    if not logfpath.exists():
        # facciamo una riga di intestazione
        with open(logfpath, 'w') as f:
            f.write("ep_len\tep_ret\n")
    # in ogni caso, dumpiamo le info correnti
    with open(logfpath, 'a') as f:   # deve essere un file di testo
        f.write("{}\t{}\n".format(episode_length, episode_return))

def do_tests(test_env, test_eps_no, max_test_episode_duration, policy, base_save_path):
    deterministic_policy = policy.create_deterministic_policy()
    for _ in range(test_eps_no):
        # reset
        obs = test_env.reset()
        episode_return = 0
        episode_duration = 0
        done = False
        while not done:
            # actions, _ = deterministic_policy.compute_actions_and_logprobs(obs)    # vediamo che succede usando il metodo che già ho
            # # è stata una brutta idea
            # act = actions[0].numpy()
            act = deterministic_policy.compute_action(obs)
            obs, rew, done, _ = test_env.step(act)
            episode_return += rew
            episode_duration += 1
            if episode_duration >= max_episode_duration:
                done=True
        simple_episode_info_dump(base_save_path/"test_ep_stats.txt", episode_duration, episode_return)




'''
PARAMETRI (in caso vogliamo farlo in modo funzionale o OOP)
env_name (per noi è sempre HumanoidStandup-v2)
seed
buffer_size
epochs
steps_per_epoch
max_episode_duration
steps_without_training
training_period
batch_size
alpha
q_lr
policy_lr
gamma
tau
save_period
test_eps_no
'''

# inizializzazione path di salvataggio
'''
ATTENZIONE!
Se [alessio] o chiunque altro usa questo codice,
deve creare un file path_constants.py e definire al suo interno
un oggetto Path in una variabile REPO_ROOT
che contenga il path assoluto della radice della repo
SULLA SUA MACCHINA.
Non carico quel file su GitHub appunto perché riguarda informazioni
specifiche della macchina usata.
'''
now = datetime.datetime.now()
unique_subdir_name = "{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}".format(
    now.year,
    now.month,
    now.day,
    now.hour,
    now.minute,
    now.second
)
base_save_path = path_constants.REPO_ROOT / "saves" / unique_subdir_name
base_save_path.mkdir()

### "Copiare" spinningup r.137~148
# Creiamo due environment: uno per il training e uno per il test
env_name = "HumanoidStandup-v2"
env = gym.make(env_name)
test_env = gym.make(env_name)
# Inizializziamo i seed a un valore fisso, per ripetibilità
seed=14383421
tf.random.set_seed(seed)
np.random.seed(seed)
env.seed(seed)
test_env.seed(seed)
# Recuperiamo le dimensioni degli spazi d'osservazione e azione (come scalari)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
# Questo potrebbe non servirci
# act_limit = env.action_space.high[0]
# Questo neppure
# ac_kwargs['action_space'] = env.action_space

### Creare 5 reti neurali: q1, q2, policy, q1_targ, q2_targ (come modelli Keras)
q1 = q_fn_module.Q_Function(env.observation_space, env.action_space)
q2 = q_fn_module.Q_Function(env.observation_space, env.action_space)
policy = policy_module.Policy(env.observation_space, env.action_space)
q1_targ = q1.create_deepcopy()
q2_targ = q2.create_deepcopy()

### Creare il replay buffer
buffer_size = 1000000
replay_buffer = replay_buffer_module.ReplayBuffer(obs_dim, act_dim, buffer_size)

# computo total_steps
epochs = 100
steps_per_epoch = 4000
total_steps = steps_per_epoch * epochs
epoch = 0

# inizializzazione environment
obs = env.reset()
episode_return = 0
episode_duration = 0

# inizializzazione optimizer
q_lr = 3e-4
q1_optimizer = tf.optimizers.Adam(learning_rate=q_lr)
q2_optimizer = tf.optimizers.Adam(learning_rate=q_lr)
policy_lr = 3e-4
policy_optimizer = tf.optimizers.Adam(learning_rate=policy_lr)

# altri parametri
warmup_steps = 10000   #per i primi 10000 passi prenderemo azioni casuali uniformemente
max_episode_duration = 1000
steps_without_training = 1000    # aspettiamo ad allenarci, in modo da riempire il buffer
training_period = 50
batch_size = 100
alpha = 0.2   # temperatura
gamma = 0.99  # discount factor
tau = 0.005   # peso per l'update delle q_targ
save_period = 10   # frequenza (in epoch) con cui salvare i modelli
test_eps_no = 10   # numero di episodi di test per epoch

# inizializzazione path
this_epoch_save_path = base_save_path / "ep{}".format(epoch)
this_epoch_save_path.mkdir()

# cronometraggio semplice
start_time = time.time()

### INIZIO LOOP PRINCIPALE
for t in range(total_steps):
    # stampa
    if t%100==0:
        print("Step {}".format(t))

    ### Ottieni la prossima azione da eseguire.
    if t > warmup_steps:
        # actions, _ = policy.compute_actions_and_logprobs(obs)    # vediamo che succede usando il metodo che già ho
        # # è stata una brutta idea
        # act = actions[0].numpy()
        act = policy.compute_action(obs)
    else:
        act = env.action_space.sample()

    ### Esegui una azione
    obs2, rew, done, _ = env.step(act)
    episode_return += rew
    episode_duration += 1

    ### il done da salvare nel replay buffer deve ignorare la terminazione causata dal tempo limite
    stored_done = False if episode_duration==max_episode_duration else done

    ### salviamo questo step nel buffer
    replay_buffer.store(obs, obs2, act, rew, stored_done)

    ### aggiornamento osservazione corrente
    obs = obs2

    ### fine episodio
    if done or episode_duration>=max_episode_duration:
        simple_episode_info_dump(this_epoch_save_path/"ep_stats.txt", episode_duration, episode_return)
        obs = env.reset()
        episode_return = 0
        episode_duration = 0

    ### Allena le reti neurali N volte ogni N step
    if t>=steps_without_training and t%training_period==0:
        for j in range(training_period):   # il rate step/train deve comunque essere 1:1, anche se facciamo gli allenamenti in "batch" piuttosto che letteralmente 1 a step
            batch = replay_buffer.random_batch(batch_size)
            # TODO pensavo di portare il training di qua, ma forse è meglio portare i modelli di là...
            #      per ora metto così, ma pensiamoci
            training_module.trainingstep_q(
                q1, q2, batch, policy, q1_targ, q2_targ,
                alpha, gamma, q1_optimizer, q2_optimizer
            )
            training_module.trainingstep_policy(policy, batch, q1, q2, alpha, policy_optimizer)
            training_module.updatestep_q_targ(q1, q2, q1_targ, q2_targ, tau)

    ### Nell'ultimo timestep di ogni epoch, potremmo voler calcolare e stampare qualche metrica, e fare altre operazioni
    if (t+1)%steps_per_epoch==0:
        print("Epoch {}/{} completata".format(epoch, epochs))
        print("    Tempo totale: {}".format(time.time()-start_time))

        epoch = (t+1) // steps_per_epoch

        # salviamo le NN ogni tanto
        if epoch%save_period==0 or epoch==epochs:  # salviamo sempre all'ultima epoch
            save_everything(this_epoch_save_path/"models",  "ep{}".format(epoch-1))
            # -1 perché adesso epoch è aggiornato all'epoch successiva,
            # ma questo salvataggio riguarda quella appena conclusa.
            # stesso motivo per cui non abbiamo ancora aggiornato
            # this_epoch_save_path.

        # facciamo dei test col modello deterministico ogni tanto
        do_tests(test_env, test_eps_no, max_episode_duration, policy, base_save_path)

        # aggiornamento path
        this_epoch_save_path = base_save_path / "ep{}".format(epoch)
        this_epoch_save_path.mkdir()

### FINE LOOP PRINCIPALE

env.close()
