# -*- coding: utf-8 -*-
"""
@created on: Thu Feb  3 14:41:59 2020
@created by: damia


Algorithm:
  1. Get data
  2. Create multiple nodes
  3. Get a arbitrary node 'local' and consider it as data owner
  4. Prepare local node to have domain encoder and a model definition
  5. Send distributed job to other workers recording input vs received data size
  6. Monitor individual training results and overall results
  7. Finally aggregate domain + aggregated and test it

"""
import numpy as np
from edil.node import SimpleWorker, SimpleProcessingNode

from edil.experiments.data_utils import get_mnist_data

from edil.th_utils import (
  SimpleTrainer, SimpleTester, SimpleImageEncoder, 
  weights_loader, weights_getter, aggregate_function
  )
from edil.th_utils import InputPlaceholder

import torch as th

class SimpleClassifier(th.nn.Module):
  def __init__(self, input_size=36, layers=[128, 64], readout=10):
    super().__init__()
    self.layers = th.nn.ModuleList()
    self.layers.append(InputPlaceholder((input_size,)))
    prev_size = input_size
    for lyr in layers:
      self.layers.append(th.nn.Linear(prev_size, lyr))
      self.layers.append(th.nn.ReLU6())
      prev_size = lyr
    self.readout_layer = th.nn.Linear(layers[-1], readout)
    return
  
  def forward(self, inputs):
    th_x = inputs
    for lyr in self.layers:
      th_x = lyr(th_x)
    th_out = self.readout_layer(th_x)
    return th_out
  
  
class TestModel(th.nn.Module):
  def __init__(self, domain_encoder, classifier):
    super().__init__()
    self.domain_encoder = domain_encoder
    self.classifier = classifier
    return
    
  def forward(self, inputs):
    th_x = self.domain_encoder(inputs)
    th_out = self.classifier(th_x)
    return th_out
  
  def predict(self, np_inputs):
    self.eval()
    dev = next(self.parameters()).device
    th_x = th.tensor(np_inputs, device=dev)
    with th.no_grad():
      th_y_hat = self(th_x)
      np_y_hat = th_y_hat.cpu().numpy()
    return np_y_hat
      
  


def load_domain_encoder():
  class DomainEncoderWrapper:
    def __init__(self, model):
      self.model = model
      self.dev = next(model.parameters()).device
      return
      
    def __call__(self, np_inputs):
      th_inputs = th.tensor(np_inputs, device=self.dev)
      th_dl = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(th_inputs),
        batch_size=512
        )
      lst_out = []
      self.model.eval()
      with th.no_grad():
        for th_batch in th_dl:
          th_out = self.model(th_batch[0])
          np_out = th_out.cpu().numpy()
          lst_out.append(np_out)
      np_out_all = np.concatenate(lst_out)
      return np_out_all  
    
  fn_enc = '_cache/mnist_enc36.pt'
  enc = SimpleImageEncoder(h=28, w=28, channels=1, scale=4)
  dev = th.device('cuda')
  enc.eval()
  enc.to(dev)
  enc.load_state_dict(th.load(fn_enc))
  de = DomainEncoderWrapper(enc)
  return enc, de

if __name__ == '__main__':
  
  (x_train, y_train), (x_dev, y_dev), (x_test, y_test) = get_mnist_data(as_numpy=True)
  
  #network simulation setup
  w1 = SimpleWorker(
    load=0.25,
    node=SimpleProcessingNode(
      name="RW1",
      )
    )
  w2 = SimpleWorker(
    load=0.50,
    node=SimpleProcessingNode(
      name="RW2",
      )
    )
  w3 = SimpleWorker(
    load=0.001,
    node=SimpleProcessingNode(
      name="LW3", # this is the local "master" worker
      )
    )
  w4 = SimpleWorker(
    load=0.25,
    node=SimpleProcessingNode(
      name="RW4",
      )
    )
  
  # we assume that we are locally on w3 node
  local = w3.node
  
  # we assume that local node already has the domain encoder (as it should in production)
  # for this experiment see ./experiments/other/ae_test.py for training of a basic 
  # mnist autoencoder and extraction of encoder part
  # below functions is just a "wrapper" around the model providing a simple call-able
  # interface
  th_model, domain_enc_func = load_domain_encoder()
  
  
  # now we distribute the training of the classifier
  fl_model = local.distributed_train(
    domain_encoder=domain_enc_func, # pass a callable model/function
    model_class=SimpleClassifier,  # class definition here
    model_weights_loader=weights_loader, # callback definition
    model_weights_getter=weights_getter, # callback definition
    train_data=(x_train, y_train), # numpy data
    dev_data=(x_dev, y_dev),  # numpy data
    test_data=(x_test, y_test),  # numpy data
    workers=[w1, w2, w4], # worker definition
    rounds=10, # number of rounds to run
    epochs_per_round=5, # epochs in each round for each worker
    train_class=SimpleTrainer, # framework based trainer class definition
    test_class=SimpleTester, # framework based tester class definition
    aggregate_fn=aggregate_function, # framework based model aggregation function
    )
  
  # now we compile domain encoder with the FL model
  dev = next(th_model.parameters()).device
  fl_model.to(dev)
  local_model = TestModel(th_model, fl_model)
  
  y_hat = local_model.predict(x_dev)
  y_pred = y_hat.argmax(1)
  dev_acc = (y_pred == y_dev).sum() / y_hat.shape[0]

  y_hat = local_model.predict(x_test)
  y_pred = y_hat.argmax(1)
  tst_acc = (y_pred == y_test).sum() / y_hat.shape[0]

  print("End-to-end dev result:  {:.4f}".format(dev_acc))
  print("End-to-end test result: {:.4f}".format(tst_acc))
  
  print("====================================================")
  print("Training classic model")
  
  enc_raw = SimpleImageEncoder(h=28, w=28, channels=1, scale=4)
  clf_raw = SimpleClassifier()
  model = TestModel(enc_raw, clf_raw)
  trainer = SimpleTrainer()
  tester = SimpleTester()
  trainer(
    model=model, 
    x_train=x_train, 
    y_train=y_train, 
    epochs=5 * 10,
    dev_func=tester,
    dev_data=(x_dev, y_dev),
    batch_size=256,
    verbose=True,
    )
  
  
  