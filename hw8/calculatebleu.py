import sys
import os
from helper import Bleu

if len(sys.argv) != 3:
  raise ValueError('Must provide two filed paths')

b = Bleu()

out = open(os.path.join(os.path.dirname(__file__), './bleu_out.txt'), 'w')
val = b.evaluate(sys.argv[1], sys.argv[2])
print val
out.write(str(val))
out.close()
