'''
Modulo che contiene le funzioni di allenamento per f.ni Q, policy, etc.

TODO [francesco]
L'esistenza di questo modulo è provvisoria, l'ho creato solo perché
devo pur mettere la funzione di allenamento policy da qualche parte.
Conviene scegliere la struttura definitiva tra un paio di giorni,
quando avremo un'idea un po' migliore di come si sta sviluppando il progetto.
Alternative ad avere un modulo dedicato al training sono:
 - Mettere le funzioni di training nel modulo principale
   (pro: possiamo accedere alle variabili locali di quel file)
   (contro: il modulo principale potrebbe gonfiarsi)
 - Mettere le funzioni di training come metodi degli oggetti Policy e <funzione q>
   (pro: approccio OOP)
   (contro: non sapremmo dove mettere l'allenamento delle q_targ e (forse) di alpha.
    Potrebbero andare nel file principale semmai.)
'''
