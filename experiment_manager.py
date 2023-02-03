import sys
import os
import numpy as np
import h5py
import json
import importlib
import mysql.connector
import REAL_DSN_config as DSN_config
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


class Experiment:
    __slots__ = ('_status', '_history', '_checkpoint_accuracy', '_checkpoint_epoch',
                 'prefix', '_experiment_id', 'batch_size',
                 'dataset_path', 'early_stopper_patience', 'input_shape',
                 'log_path', 'loss', 'metrics', 'model_base',
                 'num_classes', 'optimizer', 'saved_model_path',
                 'train_epochs', 'train_images', 'train_monitor',
                 'train_targets', 'valid_images', 'valid_targets')
    
    def _get_new_id(self):
        dsn = mysql.connector.connect(**DSN_config.local_nn_experiments_CONFIG)
        cursor = dsn.cursor()
        query = 'select max(Id) from experimentlist;'
        cursor.execute(query)
        row = cursor.fetchone()
        dsn.close()
        _id = row[0]
        if _id is None:
            _id = 0
        return(_id+1)
    
    def _read_config_from_db(self):
        _id = str(self._experiment_id)
        dsn = mysql.connector.connect(**DSN_config.local_nn_experiments_CONFIG)
        cursor = dsn.cursor()
        query = 'select Name,Value from experimentconfig where ExperimentId = ' + _id + ';'
        cursor.execute(query)
        rows = cursor.fetchall()
        dsn.close()
        
        if len(rows) == 0:
            # if dsn returns no records from db
            sys.exit('Given experiment id does not exist.')
        
        config_dict = {}
        # convert strings to integers, as neede by most models from tensorflow.keras.applications modules
        int_params = ['batch_size',
                      'early_stopper_patience',
                      'num_classes',
                      'train_epochs']
        for n,param_row in enumerate(rows):
            p_name,p_value = param_row
            if p_name in int_params:
                config_dict[p_name] = int(p_value)
            else:
                config_dict[p_name] = p_value
        return(config_dict)

    def __init__(self, prefix = None, experiment_id = None,
                 model_base = None, input_shape = None, batch_size = None,
                 num_classes = None, metrics = None, loss = None, optimizer = None,
                 train_epochs = None, train_monitor = None, early_stopper_patience = None, 
                 dataset_path = None, saved_model_path = None, log_path = None,
                 train_targets = None, train_images = None, valid_targets = None, valid_images = None):
        
        self._status = 'naive'
        self._history = None
        self._checkpoint_accuracy = None
        if prefix == None:
            self.prefix = 'NULL'
        else:
            self.prefix = str(prefix)
        
        if experiment_id == None:
            #if experiment_id is not given it is assumed to be new 
            #experiment from scratch (config will have to be loaded from json)
            #if experiment_id _is_given_ it is assumed to be repeated 
            #experiment (config will be lookedup in db)
            self._experiment_id = self._get_new_id()
            self.dataset_path = dataset_path
            self.train_targets = train_targets
            self.train_images = train_images
            self.valid_targets = valid_targets
            self.valid_images = valid_images
            self.model_base = model_base
            self.input_shape = input_shape
            if type(self.input_shape).__name__ == 'str':
                self.input_shape = ''.join(self.input_shape.split(' '))
            self.num_classes = num_classes
            self.metrics = metrics
            self.loss = loss
            self.optimizer = optimizer
            self.train_epochs = train_epochs
            self.train_monitor = train_monitor
            self.early_stopper_patience = early_stopper_patience
            self.saved_model_path = saved_model_path
            if self.saved_model_path is not None:
                self.saved_model_path = os.path.join(self.saved_model_path,'')
            self.log_path = log_path
            if self.log_path is not None:
                self.log_path = os.path.join(self.log_path,'')
            self.batch_size = batch_size
        else:
            self._experiment_id = experiment_id
            config_dict = self._read_config_from_db()
            for att in config_dict.keys():
                try:
                    setattr(self,att,config_dict[att])
                except Exception as e:
                    print(att, 'not set')
                    print(str(e))
                    
    @property
    def output_filename(self):
        return str(self.prefix) + '_' + str(self._experiment_id)
    
    @output_filename.setter
    def output_filename(self,name):
        self.prefix = name
        self._experiment_id = self._get_new_id()
    
    @output_filename.deleter
    def output_filename(self):
        pass
    
    @property
    def _cfg(self):
        config_dict = {}
        for att in dir(self):
            if not att.startswith('_'):
                config_dict[att] = self.__getattribute__(att)
        return config_dict
        
    
    @_cfg.setter
    def _cfg(self,cfg):
        for att in cfg:
            if hasattr(self,att):
                setattr(self,att,cfg[att])
            else:
                print('object has no attribute', att)
        if type(self.input_shape).__name__ == 'str':
            self.input_shape = ''.join(self.input_shape.split(' '))
    
    @_cfg.deleter
    def _cfg(self):
        pass
    
    def _import_architecture(self):
        if (type(self.input_shape).__name__ != 'str') or (type(self.num_classes).__name__ != 'int'):
            sys.exit('can not import architecture with shape and/or num_classes paratemetrs missing.')
        p, m = self.model_base.rsplit('.', 1)
        mod = importlib.import_module(p)
        met = getattr(mod,m)
        shape = list(self.input_shape.split(','))
        for n,i in enumerate(shape):
            shape[n] = int(''.join(i.split(' ')))
        try:
            model = met(weights = None,input_shape = shape,classes = self.num_classes)
            return(model)
        except Exception as e:
            print('shape =',shape,'num_classes =',self.num_classes)
            sys.exit('can not import architecture. '+ str(e))
    
    def _import_data(self):
        if self.dataset_path == None:
            sys.exit('can not import data. dataset_path parameter missing.')
        if os.path.splitext(self.dataset_path)[1] != '.h5':
            sys.exit('can not import data. wrong filetype. h5 file was expected.')
        
        keys = []
        data = []
        with h5py.File(self.dataset_path,'r') as f:
            for key in f.keys():
                data.append(f[key][:])
                keys.append(key)
        return(dict(zip(keys,data)))
    
    def _compile_model(self):
        model = self._import_architecture()
        try:
            model.compile(optimizer = self.optimizer,
                          loss = self.loss,
                          metrics = self.metrics)
        except Exception as e:
            print('optimizer =',self.optimizer,
                  'loss =',self.loss,
                  'metrics =',self.metrics)
            sys.exit('can not compile model with optimizer and/or loss and/or metrics parameters missing. '+str(e))
        return model
    
    def _train_model(self):
        #set up callbacks
        model = self._compile_model()
        #print(model.summary())
        dataset = self._import_data()
        if len(np.unique(dataset[self.train_targets])) != self.num_classes:
            sys.exit('dataset_error: self.num_classes=' + str(self.num_classes) + \
                     " doesn't match len(unique(dataset[self.train_targets]))=" + \
                     str(len(np.unique(dataset[self.train_targets]))))
        try:
            print(dataset[self.train_images].shape)
            print(dataset[self.train_targets].shape)
            print(dataset[self.valid_images].shape)
            print(dataset[self.valid_targets].shape)
        except Exception as e:
            sys.exit('dataset_error ' + type(dataset).__name__ + ' ' + str(len(dataset)) + str(e))
        
        try:
            earlystopper = EarlyStopping(monitor = self.train_monitor,
                                         patience = self.early_stopper_patience,
                                         verbose = 2)
        except Exception as e:
            sys.exit('early stopping parameters not set. ' + str(e))
            
        try:
            checkpointer = ModelCheckpoint(self.saved_model_path + self.output_filename + '.h5',
                                           verbose = 2,
                                           save_best_only = True)
        except Exception as e:
            sys.exit('model checkpointer parameters not set. ' + str(e))
            
        try:
            tbCallBack = TensorBoard(log_dir = self.log_path + self.output_filename,
                                     histogram_freq = 0,
                                     write_graph = False,
                                     write_images = False)
        except Exception as e:
            sys.exit('tensorboard parameters not set. ' + str(e))
        try:
            train_history = model.fit(dataset[self.train_images],
                                      np.arange(self.num_classes) == dataset[self.train_targets][:,None].astype(np.float32),
                                      validation_data = (dataset[self.valid_images],
                                                         np.arange(self.num_classes) == \
                                                         dataset[self.valid_targets][:,None].astype(np.float32)),
                          epochs = self.train_epochs,
                          batch_size = self.batch_size,
                          callbacks = [earlystopper,checkpointer,tbCallBack])
            self._status = 'complete'
            self._history = train_history.history
            val_loss = np.array(self._history['val_loss'])
            self._checkpoint_epoch = np.where(val_loss == val_loss.min())[0][0]
            self._checkpoint_accuracy = self._history['val_accuracy'][self._checkpoint_epoch]
        except Exception as e:
            self._status = 'failed'
            sys.exit('training failed. ' + str(e))
        
        return train_history.history
    
    def _write_config_to_db(self):
        _id = str(self._experiment_id)
        dsn = mysql.connector.connect(**DSN_config.local_nn_experiments_CONFIG)
        cursor = dsn.cursor()
        query = 'insert into experimentlist (Name,Status,Accuracy) \
        values ("' + self.prefix + str(self._experiment_id) + '","' + self._status + '",' + str(self._checkpoint_accuracy) + ');'
        try:
            cursor.execute(query)
            dsn.commit()
        except Exception as e:
            print('exception has occured in experimentlist. ' + str(e))
            dsn.rollback()
        dsn.close()
        
        dsn = mysql.connector.connect(**DSN_config.local_nn_experiments_CONFIG)
        cursor = dsn.cursor()
        for att in self._cfg:
            if type(self._cfg[att]).__name__ == 'str':
                v = '"' + str(self._cfg[att]) + '"'
            else:
                v = str(self._cfg[att])
            query = 'insert into experimentconfig (ExperimentId,Name,Value) values  (' + \
            _id + ',"' + att + '",' + v + ');'
            try:
                cursor.execute(query)
                dsn.commit()
            except Exception as e:
                print('exception has occured in experimentconfig. '+str(e))
                dsn.rollback()
        dsn.close()
        return(None)

    def _dump_history_log(self):
        with open(self.log_path + self.output_filename + '_history.txt','w') as f:
            json.dump(self._history,f)
    
