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

class SAC:
    def __init__(
        self,
        env_name = "HumanoidStandup-v2",         # nome dell'environment openaiGYM da usare
        seed = 14383421,             # seed per tutte le componenti random
        buffer_size = 1000000,      # dimensione del replay buffer
        epochs = 100,           # durata del training in epoch
        steps_per_epoch = 4000,    # durata di ogni epoch in timestep
        # le due precedenti determinano la durata totale del training
        max_episode_duration = 1000,    # durata massima di un episodio in timestep
        # le tre precedenti determinano il numero di episodi di training che verranno svolti
        warmup_steps = 10000,    # numero di timestep iniziali in cui eseguiremo azioni casuali prima di affidarci alla policy, per esplorare di più
        steps_without_training = 1000,    # numero di timestep iniziali in cui non eseguiremo training, per lasciar riempire il replay buffer
        training_period = 50,    # frequenza in timestep degli allenamenti
        batch_size = 100,     # dimensione di ciascuna batch estratta per l'allenamento
        # (viene estratta dal replay buffer, quindi scegliere buffer_size e steps_without_training per assicurarsi che esso sia abbastanza pieno)
        q_lr = 3e-4,    # learning rate per le funzioni q
        policy_lr = 3e-4,    # learning rate per la policy
        alpha = 0.2,    # parametro temperatura
        gamma = 0.99,   # parametro discount factor
        tau = 0.005,     # parametro peso per l'update delle q_targ
        save_period = 10,   # frequenza in epoch con cui salvare i modelli
        test_eps_no = 10    # numero di episodi di test da svolgere alla fine di ogni epoch
    ):
        # Creiamo due environment: uno per il training e uno per il test
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name)
        # Inizializziamo i seed a un valore fisso, per ripetibilità
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(seed)
        # Recuperiamo le dimensioni degli spazi d'osservazione e azione (come scalari)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        # Creare 5 reti neurali: q1, q2, policy, q1_targ, q2_targ (come modelli Keras)
        self.q1 = q_fn_module.Q_Function(
            self.env.observation_space, self.env.action_space
        )
        self.q2 = q_fn_module.Q_Function(
            self.env.observation_space, self.env.action_space
        )
        self.policy = policy_module.Policy(
            self.env.observation_space, self.env.action_space
        )
        self.q1_targ = self.q1.create_deepcopy()
        self.q2_targ = self.q2.create_deepcopy()
        ### Creare il replay buffer
        self.replay_buffer = replay_buffer_module.ReplayBuffer(
            self.obs_dim, self.act_dim, buffer_size
        )
        # computo total_steps
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = self.steps_per_epoch * self.epochs
        # inizializzazione optimizer
        self.q1_optimizer = tf.optimizers.Adam(learning_rate=q_lr)
        self.q2_optimizer = tf.optimizers.Adam(learning_rate=q_lr)
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=policy_lr)
        # altri parametri
        self.warmup_steps = warmup_steps
        self.max_episode_duration = max_episode_duration
        self.steps_without_training = steps_without_training
        self.training_period = training_period
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.save_period = save_period
        self.test_eps_no = test_eps_no
        # inizializzazione path di salvataggio
        '''
        ATTENZIONE!
        Se Alessio o chiunque altro usa questo codice,
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
        self.base_save_path = path_constants.REPO_ROOT / "saves" / unique_subdir_name
        self.base_save_path.mkdir()

    def go_train(self):
        # prima di cominciare, delle ultime inizializzazioni
        # epoch
        epoch = 0
        # episodio
        obs = self.env.reset()
        episode_return = 0
        episode_duration = 0
        #path
        this_epoch_save_path = self.base_save_path / "ep{}".format(epoch)
        this_epoch_save_path.mkdir()

        # cronometraggio semplice
        start_time = time.time()

        ### INIZIO LOOP PRINCIPALE
        for t in range(self.total_steps):
            # stampa
            if t%100==0:
                print("Step {}".format(t))

            ### Ottieni la prossima azione da eseguire.
            if t > self.warmup_steps:
                # actions, _ = policy.compute_actions_and_logprobs(obs)    # vediamo che succede usando il metodo che già ho
                # # è stata una brutta idea
                # act = actions[0].numpy()
                act = self.policy.compute_action(obs)
            else:
                act = self.env.action_space.sample()

            ### Esegui una azione
            obs2, rew, done, _ = self.env.step(act)
            episode_return += rew
            episode_duration += 1

            ### il done da salvare nel replay buffer deve ignorare la terminazione causata dal tempo limite
            stored_done = False if episode_duration==self.max_episode_duration else done

            ### salviamo questo step nel buffer
            self.replay_buffer.store(obs, obs2, act, rew, stored_done)

            ### aggiornamento osservazione corrente
            obs = obs2

            ### fine episodio
            if done or episode_duration>=self.max_episode_duration:
                self.simple_episode_info_dump(this_epoch_save_path/"ep_stats.txt", episode_duration, episode_return)
                obs = self.env.reset()
                episode_return = 0
                episode_duration = 0

            ### Allena le reti neurali N volte ogni N step
            if t>=self.steps_without_training and t%self.training_period==0:
                for j in range(self.training_period):   # il rate step/train deve comunque essere 1:1, anche se facciamo gli allenamenti in "batch" piuttosto che letteralmente 1 a step
                    batch = self.replay_buffer.random_batch(self.batch_size)
                    # TODO pensavo di portare il training di qua, ma forse è meglio portare i modelli di là...
                    #      per ora metto così, ma pensiamoci
                    training_module.trainingstep_q(
                        self.q1, self.q2, batch, self.policy,
                        self.q1_targ, self.q2_targ, self.alpha, self.gamma,
                        self.q1_optimizer, self.q2_optimizer
                    )
                    training_module.trainingstep_policy(
                        self.policy, batch, self.q1, self.q2,
                        self.alpha, self.policy_optimizer
                    )
                    training_module.updatestep_q_targ(
                        self.q1, self.q2, self.q1_targ, self.q2_targ, self.tau
                    )

            ### Nell'ultimo timestep di ogni epoch, potremmo voler calcolare e stampare qualche metrica, e fare altre operazioni
            if (t+1)%self.steps_per_epoch==0:
                print("Epoch {}/{} completata".format(epoch, self.epochs))
                print("    Tempo totale: {}".format(time.time()-start_time))

                epoch = (t+1) // self.steps_per_epoch

                # salviamo le NN ogni tanto
                if epoch%self.save_period==0 or epoch==self.epochs:  # salviamo sempre all'ultima epoch
                    self.save_everything(this_epoch_save_path/"models",  "_ep{}".format(epoch-1))
                    # -1 perché adesso epoch è aggiornato all'epoch successiva,
                    # ma questo salvataggio riguarda quella appena conclusa.
                    # stesso motivo per cui non abbiamo ancora aggiornato
                    # this_epoch_save_path.

                # facciamo dei test col modello deterministico ogni tanto
                self.do_tests(this_epoch_save_path)

                # aggiornamento path
                this_epoch_save_path = self.base_save_path / "ep{}".format(epoch)
                this_epoch_save_path.mkdir()

        ### FINE LOOP PRINCIPALE

        self.env.close()

    '''
    Funzioni ausiliarie
    '''

    def save_everything(self, save_path, suffix):
        # TODO devo fare __getstate__ e __setstate__
        return
        # il codice seguente è ora deprecato
        import pickle
        if not save_path.exists():
            save_path.mkdir()
        _q1_ = self.q1.create_deepcopy()
        with open(save_path/"q1{}.pkl".format(suffix), "wb") as f:
            pickle.dump(_q1_, f)
        with open(save_path/"q2{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.q2, f)
        with open(save_path/"policy{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.policy, f)
        with open(save_path/"q1targ{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.q1targ, f)
        with open(save_path/"q2targ{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.q2targ, f)
        with open(save_path/"replaybuffer{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.replay_buffer, f)
        # poi ho l'impressione che dovrei salvare i modelli separatamente, per sicurezza
        self.policy.means_and_sigmas_model.save(save_path/"policy{}.h5".format(suffix))
        self.q1.q_model.save(save_path/"q1{}.h5".format(suffix))
        self.q2.q_model.save(save_path/"q2{}.h5".format(suffix))
        self.q1targ.q_model.save(save_path/"q1targ{}.h5".format(suffix))
        self.q2targ.q_model.save(save_path/"q2targ{}.h5".format(suffix))

    # in realtà questa è praticamente "statica"
    def simple_episode_info_dump(self, logfpath, episode_length, episode_return):
        if not logfpath.exists():
            # facciamo una riga di intestazione
            with open(logfpath, 'w') as f:
                f.write("ep_len\tep_ret\n")
        # in ogni caso, dumpiamo le info correnti
        with open(logfpath, 'a') as f:   # deve essere un file di testo
            f.write("{}\t{}\n".format(episode_length, episode_return))

    def do_tests(self, base_save_path):
        deterministic_policy = self.policy.create_deterministic_policy()
        for _ in range(self.test_eps_no):
            # reset
            obs = self.test_env.reset()
            episode_return = 0
            episode_duration = 0
            done = False
            while not done:
                # actions, _ = deterministic_policy.compute_actions_and_logprobs(obs)    # vediamo che succede usando il metodo che già ho
                # # è stata una brutta idea
                # act = actions[0].numpy()
                act = deterministic_policy.compute_action(obs)
                obs, rew, done, _ = self.test_env.step(act)
                episode_return += rew
                episode_duration += 1
                if episode_duration >= self.max_episode_duration:
                    done=True
            self.simple_episode_info_dump(base_save_path/"test_ep_stats.txt", episode_duration, episode_return)



if __name__=="__main__":
    sac = SAC(epochs=1, steps_per_epoch=1000, test_eps_no=1)
    sac.go_train()
