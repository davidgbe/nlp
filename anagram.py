#!/usr/bin/python

import sys

trunc_args = sys.argv[1:]

def add_letter(curr_word, options, all_anagrams):
  if not options:
    all_anagrams.append(curr_word)
    return
  for i in range(len(options)):
    option = options[i]
    add_letter(curr_word + option, options[:i] + options[i + 1:], all_anagrams)

def run(args):
  if len(args) != 1:
    print 'You must provide only one argument'
  else:
    output_file = open('anagram_out.txt', 'w')
    anagrams = []
    add_letter('', args[0], anagrams)
    anagrams = sorted(anagrams)
    for word in anagrams:
      output_file.write(word)
    output_file.close()

run(trunc_args)
