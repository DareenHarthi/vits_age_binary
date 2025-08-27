""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_id_to_symbol_timit = {0: 'b',
 1: 'd',
 2: 'g',
 3: 'p',
 4: 't',
 5: 'k',
 6: 'bcl',
 7: 'dcl',
 8: 'gcl',
 9: 'pcl',
 10: 'tcl',
 11: 'kcl',
 12: 'jh',
 13: 'ch',
 14: 's',
 15: 'sh',
 16: 'z',
 17: 'zh',
 18: 'f',
 19: 'th',
 20: 'v',
 21: 'dh',
 22: 'hh',
 23: 'm',
 24: 'n',
 25: 'ng',
 26: 'em',
 27: 'en',
 28: 'eng',
 29: 'l',
 30: 'r',
 31: 'w',
 32: 'y',
 33: 'el',
 34: 'iy',
 35: 'ih',
 36: 'eh',
 37: 'ey',
 38: 'ae',
 39: 'aa',
 40: 'aw',
 41: 'ay',
 42: 'ah',
 43: 'ao',
 44: 'oy',
 45: 'ow',
 46: 'uh',
 47: 'uw',
 48: 'er',
 49: 'ax',
 50: 'ix',
 51: 'axr',
 52: 'iy1',
 53: 'ih1',
 54: 'eh1',
 55: 'ey1',
 56: 'ae1',
 57: 'aa1',
 58: 'aw1',
 59: 'ay1',
 60: 'ah1',
 61: 'ao1',
 62: 'oy1',
 63: 'ow1',
 64: 'uh1',
 65: 'uw1',
 66: 'er1',
 67: 'ax1',
 68: 'ix1',
 69: 'axr1',
 70: 'iy2',
 71: 'ih2',
 72: 'eh2',
 73: 'ey2',
 74: 'ae2',
 75: 'aa2',
 76: 'aw2',
 77: 'ay2',
 78: 'ah2',
 79: 'ao2',
 80: 'oy2',
 81: 'ow2',
 82: 'uh2',
 83: 'uw2',
 84: 'er2',
 85: 'ax2',
 86: 'ix2',
 87: 'axr2',
 88: 'dx',
 89: 'nx',
 90: 'q',
 91: 'hv',
 92: 'ux',
 93: 'ax-h',
 94: 'pau',
 95: 'epi',
 96: 'h#',
 97: '1',
 98: '2'}
_symbol_to_id_timit = {s: i for i, s in _id_to_symbol_timit.items()}
def text_to_sequence(text, cleaner_names):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  sequence = []
  if "timit_cleaners" in cleaner_names:
      text = text.split()
      for symbol in text:
        symbol_id = _symbol_to_id_timit[symbol]
        sequence += [symbol_id]
      return sequence
  if cleaner_names:
    clean_text = _clean_text(text, cleaner_names)
  else:
    clean_text = text 
    print(clean_text)
  for symbol in clean_text:
    symbol_id = _symbol_to_id[symbol]
    sequence += [symbol_id]
  return sequence


def cleaned_text_to_sequence(cleaned_text):
  '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
  '''
  try:
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
 
  except:
    sequence = [_symbol_to_id_timit[symbol] for symbol in cleaned_text.split()]
  return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  try:
    for symbol_id in sequence:
      s = _id_to_symbol[symbol_id]
      result += s
  except:
    
    for symbol_id in sequence:
      s = _id_to_symbol_timit[symbol_id]
      result += s   
      
  return result


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text