import tensorflow as tf
import policy as policy_module
import replay_buffer as replay_buffer_module

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
'''

### "Copiare" spinningup r.137~148
# Inizializziamo i seed a un valore fisso, per ripetibilità
seed=14383421
tf.set_random_seed(seed)
np.random.seed(seed)
# Creiamo due environment: uno per il training e uno per il test
env_name = "HumanoidStandup-v2"
env = gym.make(env_name)
test_env = gym.make(env_name)
# Recuperiamo le dimensioni degli spazi d'osservazione e azione (come scalari)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
# Questo potrebbe non servirci
# act_limit = env.action_space.high[0]
# Questo neppure
# ac_kwargs['action_space'] = env.action_space

### Creare 5 reti neurali: q1, q2, policy, q1_targ, q2_targ (come modelli Keras)
q1 = __???__
q2 = __???__
policy = policy_module.Policy(env.observation_space, env.action_space)
q1_targ = __???__
q2_targ = __???__

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
alpha = 0.2

# cronometraggio semplice
start_time = time.time()

### INIZIO LOOP PRINCIPALE
for t in range(total_steps):

    ### Ottieni la prossima azione da eseguire.
    if t > warmup_steps:
        actions, _ = policy.compute_actions_and_logprobs(obs)    # vediamo che succede usando il metodo che già ho
        act = actions[0].numpy()
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
        simple_episode_info_dump(___???___, episode_duration, episode_return)  # questo file sarebbe quello per gli episodi di training, che sarebbe DIVERSO da quello di test
        obs = env.reset()
        episode_return = 0
        episode_duration = 0

    ### Allena le reti neurali N volte ogni N step
    if t>=steps_without_training and t%training_period==0:
        for j in range(training_period):   # il rate step/train deve comunque essere 1:1, anche se facciamo gli allenamenti in "batch" piuttosto che letteralmente 1 a step
            batch = replay_buffer.random_batch(batch_size)
            q1.__???__(...)
            q2.__???__(...)
            trainingstep_policy(policy, batch, q1, q2, alpha, policy_optimizer)   # se effettivamente mettiamo il training qui dentro, ci risparmieremo molti parametri in questa funzione
            q1targ.__???__(...)
            q2targ.__???__(...)

    ### Nell'ultimo timestep di ogni epoch, potremmo voler calcolare e stampare qualche metrica, e fare altre operazioni
    if (t+1)%steps_per_epoch==0:
        print("Epoch {}/{} completata".format(epoch, epochs))
        print("    Tempo totale: {}".format(time.time()-start_time))

        epoch = (t+1) // steps_per_epoch

        # Magari salviamo le NN ogni tanto?

        # Magari facciamo dei test col modello deterministico ogni tanto?

### FINE LOOP PRINCIPALE

env.close()






def save_everything(..., suffix):
    import pickle
    from pathlib import Path
    base_path = Path(__???__)
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
    q1.__???__.save(base_path/"q1{}.h5".format(suffix))
    q2.__???__.save(base_path/"q2{}.h5".format(suffix))
    q1targ.__???__.save(base_path/"q1targ{}.h5".format(suffix))
    q2targ.__???__.save(base_path/"q2targ{}.h5".format(suffix))

def simple_episode_info_dump(logfname, episode_length, episode_return):
    # TODO sarebbe carino inizializzare logfname (da un'altra parte) a un nome di file che ancora non esiste, e crearlo vuoto (o magari con una prima riga di intestazione)
    #      magari potremmo anche fare un file diverso per ogni epoch?
    with open(logfname, 'a') as f:   # deve essere un file di testo
        f.write("{}\t{}\n".format(episode_length, episode_return))

def do_tests(...):
    deterministic_policy = policy.create_deterministic_policy()
    for _ in range(NUMEROEPISODIDITEST):
        # reset
        obs = test_env.reset()
        episode_return = 0
        episode_duration = 0
        done = False
        while not done:
            actions, _ = deterministic_policy.compute_actions_and_logprobs(obs)    # vediamo che succede usando il metodo che già ho
            act = actions[0].numpy()
            obs, rew, done, _ = test_env.step(act)
            episode_return += rew
            episode_duration += 1
            if episode_duration >= DURATAMASSIMATEST: done=True
        simple_episode_info_dump(__???__, episode_duration, episode_reward)  # questo file sarebbe quello per gli episodi di test, che sarebbe DIVERSO da quello di training
