import sys
import os
import json
import experiment_manager as em

# create empty experiemnt
# or load initial config from DB by specifying experiment_id:
# exp = em.Experiment(experiment_id = 1)
exp = em.Experiment()

with open('./REAL_cfg.json','r') as f:
    cfg = json.load(f)

# edit cfg if needed:
# cfg[attr] = foo

exp._cfg = cfg
# edit cfg via object attributes if needed:
# exp.attr_name = foo

exp.train_epochs = 1000
exp.early_stopper_patience = 50
optimizers = ['Adam','SGD','Adagrad','Adadelta']
for opt in optimizers:
    exp._experiment_id = exp._get_new_id()
    exp.optimizer = opt
    exp._train_model()
    exp._dump_history_log()
    exp._write_config_to_db()
