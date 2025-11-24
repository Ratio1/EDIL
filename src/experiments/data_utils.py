
import torch as th
import torchvision as tv

def get_mnist_data(dev=None, as_numpy=False):
  if dev is None:
    dev = th.device('cuda')
  
  train = tv.datasets.MNIST(
    root='_cache', 
    train=True, 
    transform=tv.transforms.ToTensor(),
    download=True
    )
  full_test = tv.datasets.MNIST(
    root='_cache', 
    train=False, 
    transform=tv.transforms.ToTensor(),
    download=True
    )
  

  
  x_train = (train.data.view(-1,1,28,28) / 255.).to(dev)
  y_train = train.targets.to(dev)

  n_dev = full_test.data.shape[0] // 2

  x_dev = (full_test.data[:n_dev].view(-1,1,28,28) / 255.).to(dev)
  y_dev = full_test.targets[:n_dev].to(dev)

  x_test = (full_test.data[n_dev:].view(-1,1,28,28) / 255.).to(dev)
  y_test = full_test.targets[n_dev:].to(dev)

  if as_numpy:
    return (x_train.cpu().numpy(), y_train.cpu().numpy()), (x_dev.cpu().numpy(), y_dev.cpu().numpy()), (x_test.cpu().numpy(), y_test.cpu().numpy())
  return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)
  