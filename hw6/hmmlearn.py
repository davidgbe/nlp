from lib.hmm import HiddenMarkovModel
import sys

if len(sys.argv) != 2:
  sys.exit('Must provide a path for data')

hmm = HiddenMarkovModel(sys.argv[1])
