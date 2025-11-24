# -*- coding: utf-8 -*-
"""

@created on: Tue Feb  1 08:57:01 2022
@created by: damia
"""
import numpy as np
import torch as th
import os


from edil.base import EDILBase

def th_tensor_size(t):
  return t.element_size() * t.nelement()

def th_tensor_list_size(lst):
  _size = th_tensor_size(lst[0])
  for i in range(1, len(lst)):
    is_copy = False
    t = lst[i]
    for j,t0 in enumerate(lst):
      if j != i and t0 is t:
        is_copy = True
    if not is_copy:
      _size = _size + th_tensor_size(t) 
  return _size

def th_data_size(t):
  if isinstance(t, th.Tensor):
    return t.element_size() * t.nelement()
  elif isinstance(t, list):
    return th_tensor_list_size(t)
  elif isinstance(t, th.utils.data.Dataset):
    return th_tensor_list_size(t.tensors)
  else:
    raise ValueError("Unknown data type '{}' for tensor size eval".format(t.dtype))
  return
    


def weights_loader(model, dct_np_weights):
  # loads a numpy representation of state dict for (de)serialization purposes
  keys = [k for k in dct_np_weights]
  assert isinstance(dct_np_weights[keys[0]], np.ndarray)
  dct_th_weights = {
    k : th.tensor(v.copy()) # copy to make sure we do not use reference in CPU tensor
    for k,v in dct_np_weights.items()
    }
  model.load_state_dict(dct_th_weights)
  return model


def weights_getter(model):
  # return a numpy representation of state dict as if was (de)serialized 
  is_training = model.training
  model.eval()
  with th.no_grad():
    dct_th_data = model.state_dict()
  dct_np_data = {
    k : v.cpu().numpy().copy() # copy to make sure than we dont use model reference
    for k,v in dct_th_data.items()
    }
  if is_training:
    model.train()
  return dct_np_data
  


def aggregate_state_dicts(worker_states, worker_influence, param_keys=None):
  keys = [x for x in worker_states[0]]
  if param_keys is not None: # if we know the names of the parameters
    keys = [k for k in keys if k in param_keys] # filter only required params
  assert isinstance(worker_states[0][keys[0]], np.ndarray)
  assert np.sum(worker_influence) == 1
  n_states = len(worker_states)
  # next line create zero weight including running averages and other stuff
  # so that we do not bias the new aggregated model
  # dct_agg = {k:np.zeros_like(v) for k,v in states[0].items()}
  dct_agg = {k:v.copy() for k,v in worker_states[0].items()}
  for key in keys:
    dct_agg[key] = np.zeros_like(dct_agg[key])
    for i in range(n_states):
      dct_agg[key] += worker_states[i][key] * worker_influence[i]
  return dct_agg


def th_aggregate(destionation_model, worker_states, weights):
  # TODO: check aggregation method for BN and similar layers!!!!
  param_keys = [k[0] for k in destionation_model.named_parameters()]
  if not isinstance(worker_states[0][param_keys[0]], np.ndarray):
    raise ValueError("Serialized model weights must be in dict of ndarrays")
  state = aggregate_state_dicts(
    worker_states=worker_states, 
    worker_influence=weights, 
    param_keys=param_keys
    )
  weights_loader(destionation_model, dct_np_weights=state)
  return destionation_model


def aggregate_function(original, workers, weights):
  return th_aggregate(
    destionation_model=original, 
    worker_states=workers, 
    weights=weights
    )



class SimpleTrainer(EDILBase):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return
  
  def __call__(self, **kwargs):
    self.train(**kwargs)
      
  def train(self, 
            model=None, 
            train_data=None,
            x_train=None, 
            y_train=None, 
            dev_func=None,
            dev_data=None,
            epochs=1, 
            batch_size=32,
            loss='cce', 
            optimizer='adam',
            force_gpu=True,
            verbose=None,
            ):
    assert model is not None
    verbose = self.verbose if verbose is None else verbose
    train_size = None
    train_data_np = None
    model_name =  model.__class__.__name__
    dev = next(model.parameters()).device
    if dev.type != 'cuda' and th.cuda.is_available() and force_gpu:
      dev = th.device('cuda')
      model.to(dev)
      
    dev = next(model.parameters()).device
    self.P("Starting training process for '{}' for {} epochs on {}...".format(
      model_name, epochs, dev))
    if verbose: 
      self.P("  Running training job on model '{}' device '{}'...".format(
        model_name,
        dev))
    
    if ((x_train is not None) and (y_train is not None)):
      if isinstance(x_train, np.ndarray):
        x_train = th.tensor(x_train, device=dev)
        y_train = th.tensor(y_train, device=dev)
        train_data_np = (x_train, y_train)
      th_ds = th.utils.data.TensorDataset(x_train, y_train)
      train_size = x_train.shape[0]
    elif train_data is not None and isinstance(train_data, th.utils.data.Dataset):        
      th_ds = train_data
      if isinstance(th_ds, th.utils.data.TensorDataset):
        train_size = th_ds.tensors[0].shape[0]
    else:
      raise ValueError('Please pass either x_train, y_train ndarrays or x_data torch Dataset')
    
    data_size = th_data_size(th_ds)
    self.P("  Received training dataset of size {:.2f} MB".format(data_size / 1024**2))
    
    th_ldr = th.utils.data.DataLoader(
      dataset=th_ds,
      batch_size=batch_size
      )
    
    if loss.lower() == 'cce':
      loss_fn = th.nn.CrossEntropyLoss()
    elif loss.lower() == 'mse':
      loss_fn = th.nn.MSELoss()
    else:
      raise ValueError("Unknown loss function '{}'".format(loss))
      
    if optimizer.lower() == 'adam':
      opt = th.optim.Adam(model.parameters())
    else:
      raise ValueError("Unknown optimizer '{}'".format(optimizer))    
    if train_size is not None:
      nr_batches = train_size // batch_size 
      show_step =  nr_batches // 100
    dev_results = []
    for epoch in range(1, epochs + 1):
      if verbose:
        self.P("  Epoch {}/{}...".format(epoch, epochs))
      model.train()
      losses = []
      for idx_batch, (th_x_batch, th_y_batch) in enumerate(th_ldr):
        th_yh = model(th_x_batch)
        th_loss = loss_fn(th_yh, th_y_batch)
        np_loss = th_loss.detach().cpu().numpy()
        losses.append(np_loss)
        opt.zero_grad()
        th_loss.backward()
        opt.step()
        if (train_size is not None) and (idx_batch % show_step) == 0:
          self.Pr("    Processed epoch {}/{} {:.1f}% - loss: {:.4f}".format(
            epoch, epochs,
            (idx_batch + 1) / nr_batches * 100,
            np_loss
            ))
      if verbose:
        self.P('    Epoch {} mean loss: {:.4f}{}'.format(epoch, np.mean(losses), ' ' * 50))
      if dev_func is not None and dev_data is not None:
        dev_res = dev_func(model, dev_data, verbose=verbose)
        dev_results.append(dev_res)

    if verbose and (dev_func is not None) and (train_data_np is not None):
      trn_res = dev_func(
        model=model, 
        data=train_data_np, 
        verbose=False
        )
      self.P("  Train result: {:.4f}".format(trn_res))
      self.P("  Dev result:   {:.4f}".format(dev_res))
    dct_res = model.state_dict()
    return dct_res
  
  
class SimpleTester(EDILBase):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    return
  
  def __call__(self, model, data, verbose=None, method='acc', batch_size=32, test_name='Test'):
    x_test, y_test = data
    in_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    if not isinstance(x_test, th.Tensor):
      th_x_test = th.tensor(x_test, device=dev)
      th_y_test = th.tensor(y_test, device=dev)
    else:
      th_x_test = x_test
      th_y_test = y_test
    th_dl = th.utils.data.DataLoader(
      dataset=th.utils.data.TensorDataset(
        th_x_test, th_y_test
        ),
      batch_size=batch_size
      )
    with th.no_grad():
      lst_th_yh = []
      for th_x_batch, th_y_batch in th_dl:
        th_yh_batch = model(th_x_batch)
        lst_th_yh.append(th_yh_batch)
      th_yh = th.cat(lst_th_yh)
      if method == 'acc':
        th_yp = th_yh.argmax(1)
        th_acc = (th_yp == th_y_test).sum() / th_y_test.shape[0]
      else:
        raise ValueError("Unknown testing method '{}'".format(method))
      res = th_acc.cpu().numpy()
    verbose = self.verbose if verbose is None else verbose
    if verbose:
      self.P("    {} '{}' result: {:.4f}".format(test_name, method, res))
    if in_training:
      model.train()
    return res
  
  
BASIC_ENCODER = [
  {
   "kernel"  : 3,
   "stride"  : 2,
   "filters" : 32,
   "padding" : 1
  },

  {
   "kernel"  : 3,
   "stride"  : 2,
   "filters" : 128,
   "padding" : 1,
  },
    
  {
   "kernel"  : 3,
   "stride"  : 1,
   "filters" : None, # this will be auto-calculated for last encoding layer
   "padding" : 1,
  },
  ]



class GlobalMaxPool2d(th.nn.Module):
  def __init__(self):
    super().__init__()
    return
  
  def forward(self, inputs):
    th_x = th.nn.functional.max_pool2d(
      inputs, 
      kernel_size=inputs.size()[2:]
      )
    th_x = th.squeeze(th.squeeze(th_x, -1), -1)
    return th_x
  
class ReshapeLayer(th.nn.Module):
  def __init__(self, shape):
    super().__init__()
    self._shape = shape
    return
  
  def __repr__(self):
    return self.__class__.__name__ + "{}".format(tuple([x for x in self._shape]))
  
  def forward(self, inputs):
    return inputs.view(-1, *self._shape)
  

class InputPlaceholder(th.nn.Module):
  def __init__(self, shape):
    super().__init__()
    self._shape = shape
    return
  
  def __repr__(self):
    return self.__class__.__name__ + "{}".format(tuple([x for x in self._shape]))
  
  def forward(self, inputs):
    return inputs
  
  
def calc_embed_size(h, w, c, root=3, scale=1):
  img_size = h * w * c
  v = int(np.power(img_size, 1/root))
  v = v * scale
  # now cosmetics
  vf = int(v / 4) * 4
  return vf
  
   

class SimpleImageEncoder(th.nn.Module):
  def __init__(self, h, w, channels,
               root=3, scale=1,
               layers=BASIC_ENCODER):
    super().__init__()
    self.hw = (h, w)
    self.layers = th.nn.ModuleList()
    last_channels = channels
    for layer in layers:
      k = layer.get('kernel', 3)
      s = layer.get('stride', 2)
      p = layer.get('padding', 1)
      f = layer['filters']
      if f is None:
        f = calc_embed_size(
          h, w, 
          c=channels,
          root=root,
          scale=scale
          )
      cnv = th.nn.Conv2d(
        in_channels=last_channels, 
        out_channels=f, 
        kernel_size=k,
        stride=s,
        padding=p,
        )
      last_channels = f
      bn = th.nn.BatchNorm2d(f)
      act = th.nn.ReLU()
      self.layers.append(cnv)
      self.layers.append(bn)
      self.layers.append(act)
    self.embed_layer = GlobalMaxPool2d()
    self.encoder_embed_size = last_channels
    return
  
  def forward(self, inputs):
    th_x = inputs
    for layer in self.layers:
      th_x = layer(th_x)
      # print('  ',th_x.shape)
    th_out = self.embed_layer(th_x)
    return th_out
  
  
class SimpleImageDecoder(th.nn.Module):
  def __init__(self, 
               h, w, 
               channels, 
               embed_size=None, 
               root=3,
               scale=1,
               layers=BASIC_ENCODER
               ):
    super().__init__()
    if embed_size is None:
      embed_size = calc_embed_size(
        h, w, 
        c=channels,
        root=root,
        scale=scale,
        )
    self.hw = (h, w)
    self.layers = th.nn.ModuleList()
    reduce_layers = len([x['stride'] for x in layers if x['stride'] > 1])
    input_layer = InputPlaceholder((embed_size,))
    expansion_channels = embed_size
    expansion_h = h // (2 ** reduce_layers)
    expansion_w = w // (2 ** reduce_layers)
    expansion_size = expansion_h * expansion_w * expansion_channels
    expansion_layer = th.nn.Linear(embed_size, expansion_size)
    reshape_layer = ReshapeLayer((
      expansion_channels,
      expansion_h, expansion_w
      ))
    self.layers.append(input_layer)
    self.layers.append(expansion_layer)
    self.layers.append(reshape_layer)
    last_channels = expansion_channels
    layers.reverse()
    for layer in layers:
      k = layer.get('kernel', 3)
      s = layer.get('stride', 2)
      p = layer.get('padding', 1)
      f = layer['filters']
      if f is None:
        f = embed_size
      if s == 1:
        cnv = th.nn.Conv2d(
          in_channels=last_channels, 
          out_channels=f, 
          kernel_size=k,
          stride=1,
          padding=p
          )
      else:
        cnv = th.nn.ConvTranspose2d(
          in_channels=last_channels, 
          out_channels=f, 
          kernel_size=k-1,
          stride=s,
          # padding=p
          )
      last_channels = f
      bn = th.nn.BatchNorm2d(f)
      act = th.nn.ReLU()
      self.layers.append(cnv)
      self.layers.append(bn)
      self.layers.append(act)
    self.out_layer = th.nn.Conv2d(last_channels, channels, kernel_size=1)
    return
  
  def forward(self, inputs):
    th_embed = inputs
    th_x = th_embed
    for layer in self.layers:
      th_x = layer(th_x)
      # print('  ',th_x.shape)
    th_out = self.out_layer(th_x)
    return th_out
      
    
    
class SimpleDomainAutoEncoder(th.nn.Module):
  def __init__(self, 
               h, w, channels,
               domain_name,
               save_folder='_cache',
               root=3,
               scale=1,
               layers=BASIC_ENCODER
               ):
    super().__init__()
    
    self.domain_name = domain_name
    self.save_folder = save_folder
    
    self.encoder = SimpleImageEncoder(
      h=h, w=w, 
      channels=channels,
      layers=layers,
      root=root,
      scale=scale,
      )
    
    self.decoder = SimpleImageDecoder(
      embed_size=self.encoder.encoder_embed_size, 
      h=h, w=w, 
      channels=channels,
      layers=layers
      )
    return
  
  def forward(self, inputs):
    th_x = self.encoder(inputs)
    th_out = self.decoder(th_x)
    return th_out
  
  
  def save_encoder(self, path=None):
    if path is None:
      path = os.path.join(
        self.save_folder, 
        "{}_enc{}.pt".format(
          self.domain_name,
          self.encoder.encoder_embed_size
          )
        )
    in_train = self.encoder.training
    self.encoder.eval()
    th.save(self.encoder.state_dict(), path)
    self.encoder_save_path = path
    if in_train:
      self.encoder.train()
    return
  
  def save_decoder(self, path=None):
    if path is None:
      path = os.path.join(
        self.save_folder, 
        "{}_dec{}.pt".format(
          self.domain_name,
          self.encoder.encoder_embed_size
          )
        )
    in_train = self.decoder.training
    self.decoder.eval()
    th.save(self.decoder.state_dict(), path)
    self.decoder_save_path = path
    if in_train:
      self.decoder.train()
    return
    
    
    

if __name__ == '__main__':
  import numpy as np
  H = 28
  h, w = H, H
  enc = SimpleImageEncoder(h=h, w=w, channels=3)
  print('*****************************')
  print(enc)
  th_x = th.tensor(np.random.randint(0,255, size=(2, 3, h, w)), dtype=th.float32)
  th_yh = enc(th_x)
  print(th_yh.shape)
  # ae = SimpleAutoEncoder(h=h, w=w)
  # print(ae)
  
  dec = SimpleImageDecoder(th_yh.shape[1], h, w, channels=3)
  print('*****************************')
  print(dec)
  th_im = dec(th_yh)
  print(th_im.shape)
  
  ae = SimpleDomainAutoEncoder(h=h, w=w, channels=3)
  th_im = ae(th_x)
  print('*****************************')
  print(ae)
  print(th_im.shape)
  