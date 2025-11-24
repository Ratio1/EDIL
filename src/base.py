# -*- coding: utf-8 -*-
"""

@created on: Wed Feb  2 08:12:25 2022
@created by: damia
"""

from datetime import datetime

def _P(s):
  print('\r' + s, flush=True)

def _Pr(s):
  print("\r" + s + "\r", flush=True, end='')

def get_shortname(s):
  return "".join([x for x in s if x.isupper()])


class EDILBase:
  def __init__(self, name='', use_prefix=None, verbose=True):
    self.verbose = verbose
    self.name = name
    if use_prefix is None:
      if name == '':
        use_prefix = get_shortname(self.__class__.__name__)
      else:
        use_prefix = name.upper()
    self._prefix = use_prefix
    return
  
  def _prep_str(self, s):
    s = '[{}][{}] {}'.format(
      self._prefix,
      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      s
      )
    return s


  def P(self, s):
    s = self._prep_str(s)
    _P(s)
    return


  def Pr(self, s):
    _Pr(s)
    return
  
    