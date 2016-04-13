from lib.hmm import HiddenMarkovModel
import sys
import time

start = time.time()
hmm = HiddenMarkovModel(load=True)
hmm.predict(sys.argv[1])
end = time.time()

print (end - start)
