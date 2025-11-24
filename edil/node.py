# -*- coding: utf-8 -*-
"""

@created on: Mon Jan 31 12:20:47 2022
@created by: damia
"""
import inspect
import numpy as np
from typing import List


from edil.utils import sample_shards

from edil.base import EDILBase
    

class SimpleWorker(EDILBase):
  def __init__(self, node, load, **kwargs):
    super().__init__(name=node.name, **kwargs)
    self.load = load
    self.node = node
    

class SimpleProcessingNode(EDILBase):
  
  def __init__(self, name, **kwargs):
    super().__init__(name=name, **kwargs)
    return
  
  
  def distributed_train(self, 
                        domain_encoder, # encoder that receives ndarray and returns ndarray
                        model_class, # class definition of the target model
                        model_weights_loader, # framework function for loading weights
                        model_weights_getter, # framework function for getting weights
                        train_data,  # training data tuple
                        dev_data,  # dev data tuple
                        test_data, # test data tuple
                        workers: List[SimpleWorker], # list of workers 
                        rounds, # number of rounds
                        epochs_per_round,
                        train_class, # class with __call__ or function name
                        test_class,  # class with __call__ or function name
                        aggregate_fn, # weight/models aggregation
                        ):    
    """
    The main workhorse function for distributed h-encripted training

    Parameters
    ----------
    
    domain_encoder : class
      encoder that receives ndarray and returns ndarray                        
    
    model_class : class
      class definition of the target model                       
    
    model_weights_loader : function 
      framework aware function for loading weights                        
      
    model_weights_getter : function
      framework function for getting weights                        
      
    train_data : tuple of ndarray
      training data tuple                        
      
    dev_data : TYPE
      dev data tuple                        
      
    test_data : TYPE
      test data tuple                        
      
    workers : List[SimpleWorker]
      list of workers                        
    
    rounds : TYPE
      number of trainin rounds to distribute among workers                        
    
    epochs_per_round : TYPE
      DESCRIPTION.
    
    train_class : TYPE
      class with __call__ or function name                        
      
    test_class : TYPE
      class with __call__ or function name                        
      
    aggregate_fn : TYPE
      weight/models aggregation

    Raises
    ------
    ValueError
      DESCRIPTION.

    Returns
    -------
    model : TYPE
      the trained model instance created with model_class and loaded with
      FL-based weights from all workers

    """
    # Distributed and semi-decentralized training
    self.P("Local node '{}' distributing EFL job".format(self.name))

    assert inspect.isclass(train_class)
    assert inspect.isclass(test_class)
    assert inspect.isclass(model_class)
    
    

    # first encode data using pre-trained domain encoder
    x_train, y_train = train_data
    x_dev, y_dev = dev_data
    x_test, y_test = test_data
    self.P("Using {:.2f} MB of train and dev data".format(
      sum([x.nbytes for x in [x_train, y_train, x_dev, y_dev]]) / 1024**2
      ))
    self.P("Running domain encoding...")
    enc_train = domain_encoder(x_train)
    enc_dev = domain_encoder(x_dev)
    enc_test = domain_encoder(x_test)
            
    n_train = len(enc_train)
    n_dev = len(enc_dev)
    n_test = len(enc_test)
    self.P("  Encoded train: {} obs at {:.2f} MB out of {:.2f} MB".format(
      n_train,
      enc_train.nbytes / 1024**2,
      x_train.nbytes / 1024**2,
      ))
    self.P("  Encoded dev:   {} obs at {:.2f} MB out of {:.2f} MB".format(
      n_dev,
      enc_dev.nbytes / 1024**2,
      x_dev.nbytes / 1024**2,
      ))
    self.P("  Encoded test:  {} obs at {:.2f} MB out of {:.2f} MB".format(
      n_test,
      enc_test.nbytes / 1024**2,
      x_test.nbytes / 1024**2,
      ))
        
    # get load per worker
    worker_class = SimpleWorker
    assert isinstance(workers[0], worker_class), "Workers must be '{}'".format(worker_class.__name__)
    load_per_worker = [x.load for x in workers]
    if sum(load_per_worker) != 1:
      raise ValueError("Sum of all worker loads must be 100% (1). Received {}".format(
        load_per_worker))
    
    self.P("Using {} workers with load:".format(len(load_per_worker)))
    for w in workers:
      self.P("  Worker '{}': {:.0f}%".format(w.name, w.load * 100))
    
    test_result_per_round = []
    
    # instantiate initial model
    model = model_class()
    
    # process multiple rounds of distributed training
    for idx_round in range(1, rounds+1):    
      self.P("*" * 100)
      self.P("Round #{}...".format(idx_round))
      self.P("Sampling data...")
      train_per_worker = sample_shards(load_per_worker, n_obs=n_train)
      dev_per_worker = sample_shards(load_per_worker, n_obs=n_dev)
      model_states = []
      # load weights for current round
      self.P("Getting initial weights for all workers...")
      starting_weights = model_weights_getter(model) # TODO: return file
      for i, worker in enumerate(workers):
        
        # now select only required training and dev shards
        worker_train_shard = train_per_worker[i]
        worker_x_train = enc_train[worker_train_shard]
        worker_y_train = y_train[worker_train_shard]
        worker_train_data = (worker_x_train, worker_y_train)

        worker_dev_shard = dev_per_worker[i]
        worker_x_dev = enc_dev[worker_dev_shard]
        worker_y_dev = y_dev[worker_dev_shard]
        worker_dev_data = (worker_x_dev, worker_y_dev)

        self.P("Sending job to worker '{}':".format(worker.name))
        self.P("  Train: {} obs / {:.1f} MB".format(
          worker_train_shard.shape[0], 
          sum([x.nbytes for x in worker_train_data]) / 1024**2,
          ))
        self.P("  Dev:   {} obs / {:.1f} MB".format(
          worker_dev_shard.shape[0], 
          sum([x.nbytes for x in worker_dev_data]) / 1024**2,
          ))

        # pass model, train/dev shards to worker
        worker_model_weights = worker.node.local_train(
          model_class=model_class, # send class not model
          model_weights=starting_weights, # file name
          model_weights_loader=model_weights_loader,
          model_weights_getter=model_weights_getter,
          train_data=worker_train_data,
          dev_data=worker_dev_data,
          dev_class=test_class,
          train_class=train_class,
          epochs=epochs_per_round,
          )      
        model_states.append(worker_model_weights)
      #endfor send request to each worker
      # all requests have been send now we wait for results
      # ...
      # aggregate model
      self.P("Aggregating weights...")
      model = aggregate_fn(
        original=model, 
        workers=model_states,
        weights=load_per_worker,
        ) 
      
      # test model
      if test_class is not None:
        self.P("Performing round #{} tests:".format(idx_round))
        test_func = test_class()
        _ = test_func(
          model, 
          data=(enc_train, y_train),
          test_name='Trn-set'
          )
        test_result = test_func(
          model, 
          data=(enc_test, y_test),
          test_name='Dev-set'
          )
        test_result_per_round.append(test_result)
      #endif we can test or not
    #endfor process all rounds
    return model
    
  
  def local_train(self, 
                  model_class, 
                  model_weights, 
                  model_weights_loader,
                  model_weights_getter,
                  train_data, dev_data, 
                  train_class, 
                  dev_class, 
                  epochs, 
                  batch_size=32):
    assert inspect.isclass(train_class)
    assert inspect.isclass(dev_class)
    x_train, y_train = train_data
    self.P("Node '{}' performing local training on {:.1f} MB data with {:.1f} MB dev data".format(
      self.name,
      sum([x.nbytes for x in train_data]) / 1024**2,
      sum([x.nbytes for x in dev_data]) / 1024**2,
      ))
    train_func = train_class(
      use_prefix=self.name
      )
    dev_func = dev_class()
    model = model_class()
    model = model_weights_loader(model, model_weights)
    train_func(
      model=model, 
      x_train=x_train, 
      y_train=y_train, 
      epochs=epochs,
      dev_func=dev_func,
      dev_data=dev_data,
      batch_size=batch_size,
      verbose=False,
      )
    model_weights = model_weights_getter(model)
    
    dev_res = dev_func(
      model=model,
      data=dev_data,
      verbose=False
      )
    trn_res = dev_func(
      model=model,
      data=(x_train,y_train),
      verbose=False
      )    
    
    self.P("Train result: {:.4f}".format(trn_res))
    self.P("Dev  result:  {:.4f}".format(dev_res))
    return model_weights
  
  
if __name__ == '__main__':
  import torch as th
  from edil.th_utils import aggregate_state_dicts
  
  class Base(th.nn.Module):
    def __init__(self):
      super().__init__()    
      self.l1 = th.nn.Linear(2,5)
      self.l2 = th.nn.Linear(5,2)
      
    def forward(self, inputs):
      x = self.l1(inputs)
      x = self.l2(x)
      return x
  
  
  class Adv(th.nn.Module):
    def __init__(self):
      super().__init__()
      self.m1 = Base()
      self.m2 = Base()
      self.ml = th.nn.ModuleList()
      self.ml.append(th.nn.Linear(3, 2))
      self.ml.append(th.nn.ReLU())
      self.ml.append(th.nn.Linear(2, 1))
      
    def forward(self, inputs):
      x = self.m1(inputs)
      x = self.m2(x)
      for layer in self.ml:
        x = layer(x)
      return x
  
  m1 = Adv()
  m2 = Adv()
  m3 = Adv()
  
  k1 = list(m1.state_dict().keys())[0]
  
  s1 = m1.state_dict()
  print("M1:\n{}".format(s1[k1]))
  s2 = m2.state_dict()
  print("M2:\n{}".format(s2[k1]))
  s3 = m3.state_dict()
  states = [s1,s2,s3]
  s = aggregate_state_dicts(states)
  print("AGG:\n{}".format(s[k1]))
  print("M1:\n{}".format(m1.state_dict()[k1]))
  