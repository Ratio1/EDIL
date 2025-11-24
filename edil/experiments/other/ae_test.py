# -*- coding: utf-8 -*-
"""

@description:
@created on: Tue Feb  1 18:05:37 2022
@created by: damia


Findings:
  - scaling is required for root=3
  - bigger batchsize yelds better results on root>3

"""


import torchvision as tv
import torch as th
import matplotlib.pyplot as plt

from edil.th_utils import SimpleDomainAutoEncoder, SimpleTrainer, th_data_size

def plot_grid(imgs1, imgs2, title):
  assert len(imgs1) == len(imgs2)
  fig, axs = plt.subplots(len(imgs1),2, figsize=(2,len(imgs1)))
  fig.suptitle(title)
  for i in range(len(imgs1)):
    axs[i][0].imshow(imgs1[i].squeeze())
    axs[i][0].axis('off')
    axs[i][1].imshow(imgs2[i].squeeze())
    axs[i][1].axis('off')
  plt.show()
  return

def test_func(model, test_data, tests=[1, 500], epoch=None, **kwargs):
  x_test, y_test = test_data
  th_slice = x_test[tests]
  np_slice = th_slice.detach().cpu().numpy()
  in_train = model.training
  model.eval()
  with th.no_grad():
    th_yh = model(th_slice)
    np_res = th_yh.cpu().numpy()
  title = "Epoch {}".format(epoch)
  plot_grid(np_slice, np_res, title)
  if in_train:
    model.train()
  return
  


if __name__ == '__main__':
  from edil.experiments.data_utils import get_mnist_data
  
  TRAIN_MODE = False
  TEST_MODE = True
  SCALE = 4
  
  dev = th.device('cuda')
  (x_train, x_dev), (x_dev, y_dev), (x_test, y_test) = get_mnist_data(dev)
  
  train_data = th.utils.data.TensorDataset(x_train, x_train) 
  
  if TRAIN_MODE:
    ae = SimpleDomainAutoEncoder(
      h=28, w=28, channels=1, 
      domain_name='mnist',
      scale=SCALE,
      )
    ae.to(dev)
    
    training_eng = SimpleTrainer()
    training_eng.P("Model:\n{}".format(ae))
    
    training_eng(
      model=ae, 
      train_data=train_data, 
      dev_func=test_func, dev_data=(x_dev, y_dev), 
      epochs=75,
      batch_size=200,
      loss='mse'
      )
    
    ae.save_encoder()
    training_eng.P("Saved encoder '{}'".format(ae.encoder_save_path))
    ae.save_decoder()
    training_eng.P("Saved decoder '{}'".format(ae.decoder_save_path))
  
  if TEST_MODE:
    from edil.th_utils import SimpleImageEncoder, SimpleImageDecoder
    fn_enc = '_cache/mnist_enc36.pt'
    fn_dec = '_cache/mnist_dec36.pt'
    
    enc = SimpleImageEncoder(h=28, w=28, channels=1, scale=SCALE)
    enc.eval()
    enc.to(dev)
    enc.load_state_dict(th.load(fn_enc))
    print("Loaded domain encoder with embed size {}".format(
      enc.encoder_embed_size))
    th_enc = enc(x_test)
    
    dec = SimpleImageDecoder(h=28, w=28, channels=1, scale=SCALE)
    enc.eval()
    dec.to(dev)
    dec.load_state_dict(th.load(fn_dec))
    th_dec = dec(th_enc).detach()
    
    tests = [10,100]
    gold = x_test.cpu().numpy()[tests]
    pred = th_dec.cpu().numpy()[tests]
    plot_grid(gold, pred, 'TEST')
    
    
  
  

