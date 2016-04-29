import math
import os

class Bleu(object):
    def __init__(self):
        pass

    def clean(self, sen):
        s_low = sen.lower()
        if s_low[-1] == '.':
            s_low = s_low[:-1]
        return s_low.split()

    def clean_file(self, open_file):
        return map(lambda l: self.clean(l), open_file)

    def n_grams_for_sentence(self, sen, n, concat=False):
        sen_length = len(sen)
        if n > sen_length:
            return []
        else:
            n_grams = []
            index = 0
            while n + index <= sen_length:
                gram = sen[index: n + index]
                n_grams.append(' '.join(gram) if concat else gram)
                index += 1
            return n_grams

    def produce_count_hash(self, arr):
        count_hash = {}
        for frag in arr:
            if frag in count_hash:
                count_hash[frag] += 1
            else:
                count_hash[frag] = 1
        return count_hash

    def calc_sentence_n_gram_matches(self, n, trans, actual):
        trans_n_grams = self.n_grams_for_sentence(trans, n, True)
        trans_hash = self.produce_count_hash(trans_n_grams)
        actual_n_grams = self.n_grams_for_sentence(actual, n, True)
        actual_hash = self.produce_count_hash(actual_n_grams)
        count = 0
        for gram in actual_hash:
            if not gram in trans_hash:
                continue
            else:
                actual_count = actual_hash[gram]
                trans_count = trans_hash[gram]
                if trans_count > actual_count:
                    count += actual_count
                else:
                    count += trans_count
        return count

    def calc_corpus_n_gram_precision(self, n, trans_corpus, actual_corpus):
        total = 0
        matches = 0
        for (trans, actual) in zip(trans_corpus, actual_corpus):
            total += (len(trans) + 1 - n) if len(trans) >= n else 0
            matches += self.calc_sentence_n_gram_matches(n, trans, actual)
        return float(matches) / total

    def calc_weighted_n_gram_sum(self, n, trans_corpus, actual_corpus, weights=None):
        grams = range(1, n+1)
        if weights:
            if len(weights) != n:
                raise ValueError('Weights must be of length n')
        else:
            weights = map(lambda x: float(1) / n, grams)
        summation = 0.0
        for i in grams:
            summation += (weights[i-1] * math.log(self.calc_corpus_n_gram_precision(i, trans_corpus, actual_corpus)))
        return math.exp(summation)

    def calc_brevity_penalty(self, trans_corpus, actual_corpus):
        trans_length = 0
        actual_length = 0
        for (trans, actual) in zip(trans_corpus, actual_corpus):
            trans_length += len(trans)
            actual_length += len(actual)
        if trans_length > actual_length:
            return 1.0
        else:
            return math.exp(1.0 - float(trans_length) / actual_length)

    def calc_bleu(self, trans_corpus, actual_corpus):
        return self.calc_brevity_penalty(trans_corpus, actual_corpus) * self.calc_weighted_n_gram_sum(5, trans_corpus, actual_corpus)

    def load_file(self, path):
        return open(os.path.join(os.path.dirname(__file__), path), 'r')

    def evaluate(self, candidate_path, reference_path):
        candidate = self.clean_file(self.load_file(candidate_path))
        reference = self.clean_file(self.load_file(reference_path))
        return self.calc_bleu(candidate, reference)




    

