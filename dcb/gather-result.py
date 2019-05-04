import sys
import os
import numpy as np


def isfloat(s):
  try:
    s = float(s)
    return True
  except:
    return False


all_stps = {}

for fil in sys.stdin.readlines():
  lns = open(fil.strip()).readlines()
  for i, ln in enumerate(lns):
    ln = ln.split('Cumulative reward: ')[1].split(' ')
    for k, v in zip(ln[:-1:2], ln[1::2]):
      assert not isfloat(k) and isfloat(v)
      v = float(v)
      if k not in all_stps:
        assert i == 0
        all_stps[k] = []
      if i == 0:
        all_stps[k].append([])
      all_stps[k][-1].append(v)

for k in all_stps:
  min_len = np.min([len(ak) for ak in all_stps[k]])
  buf = np.array([ak[:min_len] for ak in all_stps[k]])
  all_stps[k] = buf

np.savez('last', **all_stps)

