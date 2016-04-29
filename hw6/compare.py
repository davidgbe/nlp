import os

def load_file(path):
  return open(os.path.join(os.path.dirname(__file__), path), 'r')

preditions = load_file('./hmmoutput.txt')
actual = load_file('./hw6-grading/catalan_corpus_test_tagged.txt')

count = 0
correct = 0

for predicted_sen, actual_sen in zip(preditions, actual):
    split_pred = map(lambda x: x.split('/')[1], predicted_sen.split(' '))        
    split_actual = map(lambda x: x.split('/')[1], actual_sen.split(' '))
    for i in range(len(split_actual)):
        if split_pred[i] == split_actual[i]:
            correct += 1
        count += 1

print correct
print count
print float(correct) / count   
