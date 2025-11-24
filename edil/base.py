# -*- coding: utf-8 -*-
"""

@created on: Wed Feb  2 08:12:25 2022
@created by: damia
"""

from datetime import datetime

def _P(s):
  """
  Print a message with a carriage return and flush.

  Parameters
  ----------
  s : str
      Message to print.
  """
  print('\r' + s, flush=True)

def _Pr(s):
  """
  Print a message with a leading and trailing carriage return without newline.

  Parameters
  ----------
  s : str
      Message to print.
  """
  print("\r" + s + "\r", flush=True, end='')

def get_shortname(s):
  """
  Derive a shorthand identifier by keeping only uppercase letters.

  Parameters
  ----------
  s : str
      Source string (typically a class name).

  Returns
  -------
  str
      Concatenated uppercase letters from the input string.
  """
  return "".join([x for x in s if x.isupper()])


class EDILBase:
  def __init__(self, name='', use_prefix=None, verbose=True):
    """
    Initialize base logging helper with prefix and verbosity.

    Parameters
    ----------
    name : str, optional
      Human-readable name to prefix log messages.
    use_prefix : str, optional
      Explicit prefix override; if None, derived from `name` or class name.
    verbose : bool, optional
      When True, prints messages; stored for subclasses.
    """
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
    """
    Prepare a log string with prefix and timestamp.

    Parameters
    ----------
    s : str
      Message to wrap.

    Returns
    -------
    str
      Formatted message.
    """
    s = '[{}][{}] {}'.format(
      self._prefix,
      datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      s
      )
    return s


  def P(self, s):
    """
    Print a formatted log line with timestamp and prefix.

    Parameters
    ----------
    s : str
      Message to print.
    """
    s = self._prep_str(s)
    _P(s)
    return


  def Pr(self, s):
    """
    Print a raw message without formatting.

    Parameters
    ----------
    s : str
      Message to print.
    """
    _Pr(s)
    return
  
    
