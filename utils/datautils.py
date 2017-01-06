from __future__ import absolute_import, division

from collections import defaultdict
from scipy import sparse
import string
import re
import numpy
import os
import bz2

def deletechars(s, exclude_chars):
    ''' Fast deletion of characters from string.
    It uses a dummy translation table, and so no mapping is applied, and we
    just delete the exclude_chars characters.
    '''
    phony_translate_table = string.maketrans("","")
    return s.translate(phony_translate_table, exclude_chars)


def deletepunctuation(s):
    ''' Fast deletion of punctuation from string'''
    return deletechars(s,string.punctuation)


def tokenize(text, foldcase=True):
    ''' 
    A very cheap and easy tokenization.
    First, remove "'s". For example, "dog's" becomes "dog".
    Second, zap utf-8 chars.
    Then, remove all punctuation and, by default, fold upper and lower case words
    and then split by whitespace.
    '''

    text = re.sub(r'\'s','', text)
    s = ''.join([s for s in text if s in string.printable])

    s = str(s) # Got to convert it to str.
    s = deletepunctuation(s)

    if foldcase:
        s = s.lower()
    return s.split()


class Vocab(object):

    '''
    A simple vocabulary class
    '''

    def __init__(self, vocab):

        """
        vocab should be a list
        
        """

        word2index = {word:i for i, word in enumerate(vocab)}
        index2word = {i:word for i, word in enumerate(vocab)}

        for key in word2index:
            assert index2word[word2index[key]] == key

        for key in index2word:
            assert word2index[index2word[key]] == key

        self.vocab = vocab
        self.word2index = word2index
        self.index2word = index2word


class Cooccurrences(object):

    """
    Class for getting co-occurrence joint and conditional probabilities.
    """


    def __init__(self, filename, cache, vocabulary_list, target_words):

        self.filename = filename
        self.cache = cache
        self.vocabulary_list = vocabulary_list
        self.target_words = target_words
        self.initialized = False

    def init(self):

        if not self.initialized:

            self.texts = self.load_text(self.filename, self.cache)
            self.vocabulary = Vocab(vocab=self.vocabulary_list)
            self.get_word_counts()
            self.set_target_words(self.target_words)
            self.get_N()

            del self.texts # Same some memory
            del self.counts

            self.initialized = True

    def deinit(self):

        if self.initialized:

            del self.N
            del self.Nj

            self.initialized = False

    def load_text(self, filename, cache, verbose=False):

        local_filepath = os.path.join(cache, filename)

        basename, extension = os.path.splitext(local_filepath)

        if extension == '.bz2':

            if os.path.exists(basename):

                if verbose:
                    print('%s already exists. Skipping uncompression.' % 
                          os.path.basename(basename))

            else:

                if verbose:
                    print('Uncompressing %s to %s' % (filename, basename))

                # Read bunzipped data to tmp_data
                with bz2.BZ2File(local_filepath) as f:
                    tmp_data = f.read()

                # Write tmp_data to filename without bz2 extension
                with open(basename, 'wb') as f:
                    f.write(tmp_data)

            return open(basename).read().strip().split('\n')

        else:
            raise Exception('Expecting a bz2 file.')

    def get_word_counts_per_text(self, text):

        counts = defaultdict(int)
        for word in text.split('|'):
            if word in self.vocabulary.word2index:
                counts[word] += 1
        return counts

    def get_word_counts(self):

        self.counts = []
        for text in self.texts:
            self.counts.append(self.get_word_counts_per_text(text))

    def set_target_words(self, target_words):
           
        self.target_words = sorted(set(target_words))

        self.target_word_to_index = {word:i for i, word in enumerate(self.target_words)}

        assert all([self.target_word_to_index[self.target_words[i]] == i 
                    for i in xrange(len(self.target_words))])

    def get_N(self):

        self.N = numpy.zeros((len(self.texts), len(self.target_words)), dtype=int)

        for j, counts_text_j in enumerate(self.counts):
            for key, value in counts_text_j.iteritems():
                try:
                    self.N[j, self.target_word_to_index[key]] += value
                except KeyError:
                    pass

        self.Nj = numpy.array([sum(counts_j.values()) for counts_j in self.counts])

    def _to_dense(self):

        self.N = self.N.todense()

    def joint_frequency(self, word_k, word_l):
    
        k = self.target_word_to_index[word_k]
        l = self.target_word_to_index[word_l]
        
        return 2*(self.N[:,k] * self.N[:,l]).sum(), numpy.dot(self.Nj,(self.Nj-1))

    def marginal_frequency(self, word_k):
        
        k = self.target_word_to_index[word_k]
        
        return (2*self.N[:,k]*(self.Nj-self.N[:,k]) +
                self.N[:,k]*(self.N[:,k]-1)).sum(), numpy.dot(self.Nj,(self.Nj-1))

    def conditional_probability(self, word_k, word_l):
        
        joint, total_1 = self.joint_frequency(word_k, word_l)
        marginal, total_2 = self.marginal_frequency(word_l)
        
        assert total_1 == total_2, (total_1, total_2)
        
        return joint/marginal
