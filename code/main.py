'''
Punto d'ingresso.
Eseguire questo script da command-line.
Accetta da command line un argomento che indica
quale config usare.
'''

# import sac as sac_module   # non qui, vedi verso la fine
import argparse

# definiamo un po' di dizionari che corrispondono
# a diversi parametri per SAC ("configurazioni")

# considerata "standard" in fase di stesura,
# ma troppo impegnativa per la mia VM
# senza accesso alla scheda grafica
M_CONFIG = {
    "buffer_size": 1000000,
    "epochs": 100,
    "steps_per_epoch": 4000,
    "max_episode_duration": 1000,
    "warmup_steps": 10000,
    "steps_without_training": 1000,
    "training_period": 50,
    "batch_size": 100,
    "save_period": 10,
    "test_eps_no": 10,
    "hidden_layer_sizes": (256,256),
    "use_monitor": True,
}

# una configurazione più piccola
S_CONFIG = {
    "buffer_size": 1000000,
    "epochs": 100,
    "steps_per_epoch": 2000,
    "max_episode_duration": 1000,
    "warmup_steps": 5000,
    "steps_without_training": 1000,
    "training_period": 50,
    "batch_size": 100,
    "save_period": 10,
    "test_eps_no": 3,
    "hidden_layer_sizes": (64,64),
    "use_monitor": True,
}

# una configurazione ancora più piccola, per la mia povera VM
XS_CONFIG = {
    "buffer_size": 500000,
    "epochs": 100,
    "steps_per_epoch": 1000,
    "max_episode_duration": 1000,
    "warmup_steps": 2500,
    "steps_without_training": 1000,
    "training_period": 50,
    "batch_size": 50,
    "save_period": 10,
    "test_eps_no": 2,
    "hidden_layer_sizes": (32,32),
    "use_monitor": True,
}

# variante della precedente, un pochino più grande
VS_CONFIG = {
    "buffer_size": 500000,
    "epochs": 175,
    "steps_per_epoch": 1000,
    "max_episode_duration": 1000,
    "warmup_steps": 2500,
    "steps_without_training": 1000,
    "training_period": 50,
    "batch_size": 50,
    "save_period": 10,
    "test_eps_no": 1,
    "hidden_layer_sizes": (64,64),
    "use_monitor": True,
}

# ulteriore variante
VS128_CONFIG = {
    "buffer_size": 500000,
    "epochs": 175,
    "steps_per_epoch": 1000,
    "max_episode_duration": 1000,
    "warmup_steps": 8000,
    "steps_without_training": 1000,
    "training_period": 50,
    "batch_size": 50,
    "save_period": 10,
    "test_eps_no": 1,
    "hidden_layer_sizes": (128,128),
    "use_monitor": True,
}

# una configurazione stupidamente piccola, a solo scopo di test
T_CONFIG = {
    "epochs": 1,
    "steps_per_epoch": 1000,
    "max_episode_duration": 500,
    "test_eps_no": 1,
    "warmup_steps": 800,
    "steps_without_training": 800,
    "use_monitor": True,
}

# un dizionario unico per selezionare una configurazione
# in base al suo codice
CONFIGS = {
    "M": M_CONFIG,
    "S": S_CONFIG,
    "XS": XS_CONFIG,
    "VS": VS_CONFIG,
    "VS128": VS128_CONFIG,
    "T": T_CONFIG,
}

# Gestione command-line
parser = argparse.ArgumentParser(description="A SAC implementation.")
parser.add_argument(
    "config",
    help='Select config. '
         "Possible config choices right now are M and S, "
         "but check CONFIGS in the source code to be sure"
)
parser.add_argument(
    '--dont-record',
    action='store_true',
    help="Specify to exclude recording (and save some time)"
)
parser.add_argument(
    '--dont-save',
    action='store_true',
    help="Specify to exclude intermediate saving. Might save some memory usage."
)
args = parser.parse_args()
config = args.config.upper()

# selezione configurazione
sac_args = CONFIGS[config]
if args.dont_record:
    sac_args["use_monitor"] = False
if args.dont_save:
    sac_args["save_period"] = 2 * sac_args["epochs"]    # equivale a "mai"

# importiamo sac all'ultimo momento,
# così se sbagliamo la command line
# non dobbiamo sorbirci il caricamento
# di tensorflow
import sac as sac_module

sac = sac_module.SAC(**sac_args)
sac.go_train()
