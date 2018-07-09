"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

import os
import csv
import re
import argparse
from six.moves import xrange
from six.moves import cPickle
from collections import defaultdict

_WORD_SPLIT = re.compile("([.,!?\";)(])")

def precook(s, n=4, out=False):
  """
  Takes a string as input and returns an object that can be given to
  either cook_refs or cook_test. This is optional: cook_refs and cook_test
  can take string arguments as well.
  :param s: string : sentence to be converted into ngrams
  :param n: int    : number of ngrams for which representation is calculated
  :return: term frequency vector for occuring ngrams
  """
  words = s.split()
  counts = defaultdict(int)
  for k in xrange(1,n+1):
    for i in xrange(len(words)-k+1):
      ngram = tuple(words[i:i+k])
      counts[ngram] += 1
  return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
  crefs = []
  for ref in refs:
    # ref is a list of 5 captions
    crefs.append(cook_refs(ref))
  return crefs

def compute_doc_freq(crefs):
  '''
  Compute term frequency for reference data.
  This will be used to compute idf (inverse document frequency later)
  The term frequency is stored in the object
  :return: None
  '''
  document_frequency = defaultdict(float)
  for refs in crefs:
    # refs, k ref captions of one image
    for ngram in set([ngram for ref in refs for (ngram,count) in ref.items()]):
      document_frequency[ngram] += 1
      # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
  return document_frequency

def get_w2i_i2w(dict_file):
  f = open(dict_file,'r')
  w2i, i2w = {}, {}
  for line in f.readlines():
    id, word, _ = line.strip().split(' ')
    w2i[word] = id
    i2w[id] = word

  return w2i, i2w

def get_ngram_words_idxs(file_name, w2i):
    count = 0
    refs_words = []
    refs_idxs = []
    for line_count, sp in enumerate(csv.reader(open(file_name))):
        if line_count == 0:
            continue
        # InputStoryid    InputSentence1  InputSentence2  InputSentence3
        # InputSentence4  RandomFifthSentenceQuiz1
        # RandomFifthSentenceQuiz2        AnswerRightEnding
        sp = [s.strip() for s in sp]
        ending = sp[-1]
        ending_list = []
        for token_punc in ending.strip().split(' '):
          words = re.split(_WORD_SPLIT, token_punc)
          w_list = [x.lower() for x in words if x]
          ending_list.extend(w_list)

        ending_str = ' '.join(ending_list)
        ending_idx = [str(w2i[w]) for w in ending_list]
        ending_idx_str = ' '.join(ending_idx)
        refs_words.append([ending_str])
        refs_idxs.append([ending_idx_str])
        count += 1
    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))

    return ngram_words, ngram_idxs, count

def main(params):

  w2i,i2w = get_w2i_i2w(params['dict'])
  print('finished getting w2i,i2w')
  ngram_words, ngram_idxs, ref_len = get_ngram_words_idxs(params['input_data'], w2i)
  print('finished getting ngram_words, ngram_idxs, ref_len')

  cPickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl']+'-words.p','wb'), protocol=cPickle.HIGHEST_PROTOCOL)
  print('finished getting doc_words pkl')
  cPickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_pkl']+'-idxs.p','wb'), protocol=cPickle.HIGHEST_PROTOCOL)
  print('finished getting doc_idxs pkl')

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_data', default='ROCStories_all.csv', help='train data')
  parser.add_argument('--dict', default='id_word_senti_score.txt', help='dict')
  parser.add_argument('--output_pkl', default='ROC_train_ending', help='output pickle file')
  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  main(params)
