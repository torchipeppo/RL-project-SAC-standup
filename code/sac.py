import numpy as np
import tensorflow as tf
import gym
import time
import datetime
import pickle
from pathlib import Path
import policy as policy_module
import q_function as q_fn_module
import replay_buffer as replay_buffer_module
import agent as agent_module
import path_constants

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
        test_eps_no = 10,    # numero di episodi di test da svolgere alla fine di ogni epoch
        hidden_layer_sizes = (256,256),   # (numero e) dimensioni delle layer nascoste di tutti i modelli
        use_monitor = True      # decide se usare il monitor di gym o no. Il monitor registra alcuni episodi e scrive altre statistiche.
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
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        # le 5 NN e il repley buffer sono stati spostati nel nuovo modulo:
        self.the_agent = agent_module.Agent(
            self.env.observation_space, self.env.action_space,
            hidden_layer_sizes,
            q_lr, policy_lr, alpha, gamma, tau
        )
        # Creare il replay buffer
        self.replay_buffer = replay_buffer_module.ReplayBuffer(
            obs_dim, act_dim, buffer_size
        )
        # computo total_steps
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = self.steps_per_epoch * self.epochs
        # altri parametri
        self.warmup_steps = warmup_steps
        self.max_episode_duration = max_episode_duration
        self.steps_without_training = steps_without_training
        self.training_period = training_period
        self.batch_size = batch_size
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
        # prepariamo i monitor se richiesti
        if use_monitor:
            monitor_path = self.base_save_path / "monitor"
            self.env = gym.wrappers.Monitor(
                self.env,
                directory=monitor_path,
                video_callable=capped_quadratic_video_schedule,
                force=True
            )
            test_monitor_path = self.base_save_path / "test_monitor"
            self.test_env = gym.wrappers.Monitor(
                self.test_env,
                directory=test_monitor_path,
                video_callable=capped_quadratic_video_schedule,
                force=True
            )

    def go_train(self):
        # prima di cominciare, delle ultime inizializzazioni
        # epoch
        epoch = 0
        # episodio
        obs = self.env.reset()
        episode_return = 0
        episode_duration = 0
        # statistiche delle NN
        q1_avg_loss = 0
        q2_avg_loss = 0
        policy_avg_loss = 0
        number_of_samples_in_avg_losses = 0
        #path
        this_epoch_save_path = self.base_save_path / "ep{}".format(epoch)
        this_epoch_save_path.mkdir()

        # cronometraggio semplice
        start_time = time.time()

        ### INIZIO LOOP PRINCIPALE
        for t in range(self.total_steps):
            # stampa
            if t%100==0:
                print("Epoch {}/{} Step {}/{}".format(epoch, self.epochs, t, self.total_steps))

            ### Ottieni la prossima azione da eseguire.
            if t > self.warmup_steps:
                # actions, _ = policy.compute_actions_and_logprobs(obs)    # vediamo che succede usando il metodo che già ho
                # # è stata una brutta idea
                # act = actions[0].numpy()
                act = self.the_agent.compute_action(obs)
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
                    # le funzioni di training sono prerogativa dell'Agent
                    loss_dict = self.the_agent.trainingstep(batch)
                    # aggiornamento statistiche (update online della media aritmetica)
                    number_of_samples_in_avg_losses += 1
                    # (usiamo i backslash backslash per andare a capo)        →→→→→→→→↓
                    q1_avg_loss += (loss_dict[agent_module.Q1_LOSS] - q1_avg_loss) /  \
                                         number_of_samples_in_avg_losses
                    q2_avg_loss += (loss_dict[agent_module.Q2_LOSS] - q2_avg_loss) /  \
                                         number_of_samples_in_avg_losses
                    policy_avg_loss += (loss_dict[agent_module.POLICY_LOSS] - policy_avg_loss) /  \
                                               number_of_samples_in_avg_losses

            ### Nell'ultimo timestep di ogni epoch, potremmo voler calcolare e stampare qualche metrica, e fare altre operazioni
            if (t+1)%self.steps_per_epoch==0:
                # Epoch completata

                # Salviamo le perdite medie di questa epoch
                self.simple_loss_info_dump(
                    this_epoch_save_path/"losses.txt",
                    policy_avg_loss,
                    q1_avg_loss,
                    q2_avg_loss
                )
                # Importante: poi resettiamo tali statistiche
                q1_avg_loss = 0
                q2_avg_loss = 0
                policy_avg_loss = 0
                number_of_samples_in_avg_losses = 0

                print("Epoch {}/{} completata".format(epoch, self.epochs))

                epoch = (t+1) // self.steps_per_epoch

                # salviamo le NN ogni tanto
                if epoch%self.save_period==0 or epoch==self.epochs:  # salviamo sempre all'ultima epoch
                    print("    Salvataggio in corso...")
                    # self.save_everything(this_epoch_save_path/"models",  "_ep{}".format(epoch-1))
                    # -1 perché adesso epoch è aggiornato all'epoch successiva,
                    # ma questo salvataggio riguarda quella appena conclusa.
                    # stesso motivo per cui non abbiamo ancora aggiornato
                    # this_epoch_save_path.
                    ########
                    # mi sono accorto che devo salvare solo i modelli più recenti,
                    # altrimenti esaurisco lo spazio su disco
                    self.save_everything(self.base_save_path/"models",  "")
                    # faccio in modo di dare lo stesso nome, così open("wb")
                    # sovrascrive

                # facciamo dei test col modello deterministico ogni tanto
                self.do_tests(this_epoch_save_path)

                # aggiornamento path
                this_epoch_save_path = self.base_save_path / "ep{}".format(epoch)
                this_epoch_save_path.mkdir()

                # tempo totale di questa epoch
                print("    Tempo totale: {}".format(time.time()-start_time))

        ### FINE LOOP PRINCIPALE

        self.env.close()
        self.test_env.close()

    '''
    Funzioni ausiliarie
    '''

    def save_everything(self, save_path, suffix):
        if not save_path.exists():
            save_path.mkdir()
        with open(save_path/"replaybuffer{}.pkl".format(suffix), "wb") as f:
            pickle.dump(self.replay_buffer, f)
        self.the_agent.save_all_models(save_path, suffix)

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
        # qui conviene accedere direttamente all'Agent
        deterministic_policy = self.the_agent.policy.create_deterministic_policy()
        for e in range(self.test_eps_no):
            print("    Episodio di test {}/{}...".format(e, self.test_eps_no))
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

    def simple_loss_info_dump(self, logfpath, policy_loss, q1_loss, q2_loss):
        if not logfpath.exists():
            # facciamo una riga di intestazione
            with open(logfpath, 'w') as f:
                f.write("policy\tq1\tq2\n")
        # in ogni caso, dumpiamo le info correnti
        with open(logfpath, 'a') as f:   # deve essere un file di testo
            f.write("{}\t{}\t{}\n".format(policy_loss, q1_loss, q2_loss))

# se passata al Monitor come argomento video_callable,
# registra ogni episodio quadrato anziché ogni cubo.
def capped_quadratic_video_schedule(episode_id):
    if episode_id < 1000:
        return int(round(episode_id ** (1. / 2))) ** 2 == episode_id
    else:
        return episode_id % 1000 == 0



if __name__=="__main__":
    sac = SAC(epochs=1, steps_per_epoch=1000, test_eps_no=1)
    sac.go_train()
