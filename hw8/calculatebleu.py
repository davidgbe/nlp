import sys
from helper import Bleu

if len(sys.argv) != 3:
  raise ValueError('Must provide two filed paths')

b = Bleu()

print b.evaluate(sys.argv[1], sys.argv[2])
