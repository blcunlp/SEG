# coding: utf-8
from collections import OrderedDict
from six.moves import xrange
import re
import os
import csv
import numpy as np
import tensorflow as tf
from metrics.bleu.bleu import Bleu
from metrics.rouge.rouge import Rouge
from metrics.cider.ciderD import CiderD
from metrics.cider.cider import Cider

_WORD_SPLIT = re.compile("([.,!?\";)(])")
_START_VOCAB = ['PAD', 'UNK', 'BOS']

BOS_ID = 2
UNK_ID = 1
PAD_ID = 0 #also as EOS
#for ROCstory
def extract_story(file_name, train=True):
    dataset = []
    for line_count, sp in enumerate(csv.reader(open(file_name))):
        if line_count == 0: # ignore the first title line
            continue
        sp = [s.strip() for s in sp]
        if train:
            idx, sentences, title = sp[0], sp[2:7], sp[1]
            dataset.append(sentences)
        else:
            idx, sentences, answer = sp[0], sp[1:5], sp[7]
            if answer == '1':
                sentences.append(sp[5])
            else:
                sentences.append(sp[6])
            dataset.append(sentences)
    return dataset

def prepare_dataset(data_path):
    train_file = os.path.join(data_path,'ROCStories_all.csv')
    val_file = os.path.join(data_path,'cloze_test_ALL_val.csv')
    test_file = os.path.join(data_path,'cloze_test_ALL_test.csv')

    train_data = extract_story(train_file, train=True)
    valid_data = extract_story(val_file, train=False)
    test_data = extract_story(test_file, train=False)

    return train_data, valid_data, test_data

def creat_vocab(data,vocab_size):
    word_counts = OrderedDict()
    max_plot_len = 0
    max_ending_len = 0
    min_ending_len = None
    for idx, sentences in enumerate(data):
        plot = []
        ending = []
        for i in xrange(len(sentences)):
            sent = []
            for token_punc in sentences[i].split(' '):
                words = re.split(_WORD_SPLIT,token_punc)
                l = [x.lower() for x in words if x]
                sent.extend(l)
            if i <= 3:
                plot.extend(sent)
            else:
                ending.extend(sent)

            for w in sent:
                if w not in word_counts:
                    word_counts[w] = 1
                else:
                    word_counts[w] += 1

        plot_len = len(plot)
        ending_len = len(ending)
        if plot_len >= max_plot_len:
            max_plot_len = plot_len
        if ending_len >= max_ending_len:
            max_ending_len = ending_len
        if min_ending_len == None or ending_len <= min_ending_len:
            min_ending_len = ending_len
    sorted_vocab = sorted(word_counts.items(), key=lambda x : x[1], reverse=True)
    vocab_list = _START_VOCAB + [w for (w,count) in sorted_vocab]
    if vocab_list != None:
        vocab_list = vocab_list[:vocab_size]
    word2id = {x:i for i,x in enumerate(vocab_list)}
    id2word = {i:x for i,x in enumerate(vocab_list)}
    return word2id,id2word, max_plot_len, max_ending_len, min_ending_len

def get_batches(n, batch_size):
    num_batches = int(np.floor(n / float(batch_size)))
    return [(i * batch_size, min(n, (i + 1) * batch_size))
                    for i in range(0, num_batches)]

def prepare_batch_data(data, start_id, end_id, word2id, max_ending_len, pointer_gen):
    batch_data = data[start_id: end_id]
    batch_plot_ids, batch_ending_ids, batch_ext_plot_ids, batch_plot_oovs = [], [], [], []
    for idx, sentences in enumerate(batch_data):
        sentences_word = []
        for s in sentences[:5]:
            sent_words = []
            for token_punc in s.split(' '):
                words = re.split(_WORD_SPLIT, token_punc)
                l = [x.lower() for x in words if x]
                sent_words.extend(l)
            sentences_word.append(sent_words)
        plot_words = []
        for i_for_sum in range(4):
            plot_words.extend(sentences_word[i_for_sum])
        ending_words = sentences_word[4]
        #get plot_ids for both seq2seq+attn and pointer-gen
        plot_ids = []
        for p_w in plot_words:
            if p_w in word2id.keys():
                p_id = word2id[p_w]
            else:
                p_id = word2id['UNK']
            plot_ids.append(p_id)

        if not pointer_gen:
            ending_ids = []
            for e_w in ending_words:
                if e_w in word2id.keys():
                    e_id = word2id[e_w]
                else:
                    e_id = word2id['UNK']
                ending_ids.append(e_id)
        else:
            ext_plot_ids = []
            plot_oovs = []
            for w in plot_words:
                if w not in word2id.keys(): # If w is OOV
                    if w not in plot_oovs:  # Add to list of OOVs
                        plot_oovs.append(w)
                    oov_num = plot_oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                    ext_plot_ids.append(len(word2id.keys()) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
                else:
                    i = word2id[w]
                    ext_plot_ids.append(i)

            ending_ids = []
            for w in ending_words:
                if w not in word2id.keys():  # If w is an OOV word
                    if w in plot_oovs:  # If w is an in-article OOV
                        vocab_idx = len(word2id.keys()) + plot_oovs.index(w)  # Map to its temporary article OOV number
                        ending_ids.append(vocab_idx)
                    else:  # If w is an out-of-article OOV
                        ending_ids.append(UNK_ID)  # Map to the UNK token id
                else:
                    ending_ids.append(word2id[w])

        batch_plot_ids.append(plot_ids)
        batch_ending_ids.append(ending_ids)
        if not pointer_gen:
            batch_ext_plot_ids = None
            batch_plot_oovs = None
        else:
            batch_ext_plot_ids.append(ext_plot_ids)
            batch_plot_oovs.append(plot_oovs)
    if pointer_gen:
        lengths_plot_oovs = [len(s) for s in batch_plot_oovs]
        max_plot_oovs = np.max(np.array(lengths_plot_oovs))
    else:
        max_plot_oovs = None
    lengths_plot = [len(s) for s in batch_plot_ids]
    max_plot_len = max(lengths_plot)
    lengths_plot = [len(s) if len(s) <= max_plot_len else max_plot_len for s in batch_plot_ids]
    lengths_ending = [len(s) if len(s) <= max_ending_len else max_ending_len for s in batch_ending_ids]

    plot = tf.keras.preprocessing.sequence.pad_sequences(batch_plot_ids, maxlen=max_plot_len, dtype='int32',
                                                         padding='post', truncating='pre', value=PAD_ID)
    ending = tf.keras.preprocessing.sequence.pad_sequences(batch_ending_ids, maxlen=max_ending_len, dtype='int32',
                                                           padding='post', truncating='pre', value=PAD_ID)
    plot_mask = np.zeros_like(plot, dtype=float)
    plot_mask[plot != PAD_ID] = 1
    if pointer_gen:
        ext_plot = tf.keras.preprocessing.sequence.pad_sequences(batch_ext_plot_ids, maxlen=max_plot_len, dtype='int32',
                                                                 padding='post', truncating='pre',value=PAD_ID)
    else:
        ext_plot = None
    return plot, ending, ext_plot, np.array(lengths_plot), np.array(lengths_ending), max_plot_oovs, plot_mask

def prepare_data_for_beam_seach_decode(data, batch_size, word2id, max_ending_len, pointer_gen):
    examples_list = []
    for idx, sentences in enumerate(data):
        sentences_word = []
        for s in sentences:
            sent_words = []
            for token_punc in s.split():
                words = re.split(_WORD_SPLIT, token_punc)
                l = [x.lower() for x in words if x]
                sent_words.extend(l)
            sentences_word.append(sent_words)
        plot_words = []
        for i_for_sum in range(4):
            plot_words.extend(sentences_word[i_for_sum])
        ending_words = sentences_word[4]
        #get plot_ids for both seq2seq+attn and pointer-gen
        plot_ids = []
        for p_w in plot_words:
            if p_w in word2id.keys():
                p_id = word2id[p_w]
            else:
                p_id = word2id['UNK']
            plot_ids.append(p_id)

        if not pointer_gen:
            ending_ids = []
            for e_w in ending_words:
                if e_w in word2id.keys():
                    e_id = word2id[e_w]
                else:
                    e_id = word2id['UNK']
                ending_ids.append(e_id)
        else:
            ext_plot_ids = []
            plot_oovs = []
            for w in plot_words:
                if w not in word2id.keys(): # If w is OOV
                    if w not in plot_oovs:  # Add to list of OOVs
                        plot_oovs.append(w)
                    oov_num = plot_oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                    ext_plot_ids.append(len(word2id.keys()) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
                else:
                    i = word2id[w]
                    ext_plot_ids.append(i)

            ending_ids = []
            for w in ending_words:
                if w not in word2id.keys():  # If w is an OOV word
                    if w in plot_oovs:  # If w is an in-article OOV
                        vocab_idx = len(word2id.keys()) + plot_oovs.index(w)  # Map to its temporary article OOV number
                        ending_ids.append(vocab_idx)
                    else:  # If w is an out-of-article OOV
                        ending_ids.append(UNK_ID)  # Map to the UNK token id
                else:
                    ending_ids.append(word2id[w])
        lengths_plot = [len(plot_ids) for _ in xrange(batch_size)]
        max_plot_len = max(lengths_plot)
        lengths_plot = [len(s) if len(s) <= max_plot_len else max_plot_len for _ in xrange(batch_size)]
        lengths_ending = [len(ending_ids) if len(plot_ids) <= max_ending_len else max_ending_len for _ in
                          xrange(batch_size)]
        batch_plot_ids = [plot_ids for _ in xrange(batch_size)]
        batch_ending_ids = [ending_ids for _ in xrange(batch_size)]
        padded_batch_plot_ids = tf.keras.preprocessing.sequence.pad_sequences(batch_plot_ids, maxlen=max_plot_len,
                                                                              dtype='int32', padding='post',truncating='pre', value=PAD_ID)
        padded_batch_ending_ids = tf.keras.preprocessing.sequence.pad_sequences(batch_ending_ids, maxlen=max_ending_len,
                                                                                dtype='int32', padding='post',truncating='pre', value=PAD_ID)
        plot_mask = np.zeros_like(padded_batch_plot_ids, dtype=float)
        plot_mask[padded_batch_plot_ids != PAD_ID] = 1
        if not pointer_gen:
            padded_batch_ext_plot_ids = [None for _ in xrange(batch_size)]
            batch_plot_oovs = [None for _ in xrange(batch_size)]
            max_plot_oovs = None
        else:
            batch_ext_plot_ids = [ext_plot_ids for _ in xrange(batch_size)]
            padded_batch_ext_plot_ids = tf.keras.preprocessing.sequence.pad_sequences(batch_ext_plot_ids,
                                                                                      maxlen=max_plot_len,
                                                                                      dtype='int32', padding='post',
                                                                                      truncating='pre', value=PAD_ID)
            batch_plot_oovs = [plot_oovs for _ in xrange(batch_size)]
            max_plot_oovs = len(plot_oovs)

        ex = (padded_batch_plot_ids, padded_batch_ending_ids, padded_batch_ext_plot_ids, np.array(lengths_plot),
              np.array(lengths_ending), max_plot_oovs, batch_plot_oovs, plot_mask)
        examples_list.append(ex)
    return examples_list

#for rl_train
def mask_base_samp_tatget_sequences(base_sequences, sample_sequences, batch_ending, batch_len_ending):
    batch_base_seqs, batch_sample_seqs, batch_target_seqs = [],[],[]
    for base_seq, sample_seq, target_seq, target_seq_len in zip(base_sequences, sample_sequences, batch_ending, batch_len_ending):
        base_seq = base_seq.tolist()
        sample_seq = sample_seq.tolist()
        if PAD_ID not in base_seq:
            masked_base_seq = base_seq
        else:
            if base_seq[0] == PAD_ID:
                new_base_seq = base_seq[1:]
                if PAD_ID not in new_base_seq:
                    masked_base_seq = [PAD_ID] + new_base_seq
                else:
                    masked_base_seq = base_seq[:new_base_seq.index(PAD_ID)+1]
            else:
                masked_base_seq = base_seq[:base_seq.index(PAD_ID)]
        if PAD_ID not in sample_seq:
            masked_sample_seq = sample_seq
        else:
            if sample_seq[0] == PAD_ID:
                new_sample_seq = sample_seq[1:]
                if PAD_ID not in new_sample_seq:
                    masked_sample_seq = [PAD_ID] + new_sample_seq
                else:
                    masked_sample_seq = sample_seq[:new_sample_seq.index(PAD_ID)+1]
            else:
                masked_sample_seq = sample_seq[:sample_seq.index(PAD_ID)]

        masked_target = target_seq[:target_seq_len]
        batch_base_seqs.append(masked_base_seq)
        batch_sample_seqs.append(masked_sample_seq)
        batch_target_seqs.append(masked_target)
    return batch_base_seqs, batch_sample_seqs, batch_target_seqs
#for rl_train
def compute_bleu_reward(base_sequences, sample_sequences, target_sequences, n_gram):
    r_bleu = []
    bleu_obj = Bleu(4)
    for base_seq, sample_seq, target_seq in zip(base_sequences, sample_sequences, target_sequences):
        target_str_ = ' '.join('%s' % id for id in target_seq)
        target_str = {0:[target_str_]}

        base_seq_str_ = ' '.join('%s' % id for id in base_seq)
        base_seq_str = {0:[base_seq_str_]}

        sample_seq_str_ = ' '.join('%s' % id for id in sample_seq)
        sample_seq_str = {0:[sample_seq_str_]}

        one_base_bleu, _, = bleu_obj.compute_score(target_str, base_seq_str)
        one_sample_bleu, _, = bleu_obj.compute_score(target_str, sample_seq_str)
        one_r_bleu = one_sample_bleu[n_gram-1] - one_base_bleu[n_gram-1]
        r_bleu.append(one_r_bleu)
    return r_bleu
#for rl_train
def compute_rouge_or_cider_reward(base_sequences, sample_sequences, target_sequences, metric, pkl_file):
    r = []
    if metric == 'rouge':
        metric_obj = Rouge()
    elif metric == 'cider':
        # assert pkl_file is not None, 'pkl_file can not be None!'
        # metric_obj = CiderD(pkl_file = pkl_file)
        metric_obj = Cider()

    for base_seq, sample_seq, target_seq in zip(base_sequences, sample_sequences,target_sequences):
        target_str_ = ' '.join('%s' % id for id in target_seq)
        target_str = {0:[target_str_]}

        base_seq_str_ = ' '.join('%s' % id for id in base_seq)
        base_seq_str = {0:[base_seq_str_]}

        sample_seq_str_ = ' '.join('%s' % id for id in sample_seq)
        sample_seq_str = {0:[sample_seq_str_]}
        one_base_score, _, = metric_obj.compute_score(target_str, base_seq_str)
        one_sample_score, _, = metric_obj.compute_score(target_str, sample_seq_str)
        one_r_bleu = one_sample_score - one_base_score
        r.append(one_r_bleu)
    return r

def result_ids2words(id_list, id2word, plot_oovs):
  """Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

  Args:
    id_list: list of ids (integers)
    vocab: Vocabulary object
    article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)

  Returns:
    words: list of words (strings)
  """
  words = []
  if plot_oovs is not None and len(plot_oovs) != 0:
    for i in id_list:
      if i in range(len(id2word.keys()), len(id2word.keys()) + len(plot_oovs)):
          oov_idx = i - len(id2word.keys())
          w = plot_oovs[oov_idx]
      else:
          w = id2word[i]
      words.append(w)
  else:
      for i in id_list:
        w = id2word [i]
        words.append(w)
  return words







if __name__ == '__main__':
    train_data, valid_data, test_data = prepare_dataset('./data/ROC_data')

    print('TrainData Size:', len(train_data))
    print('ValidData Size:', len(valid_data))
    print('TestData Size:', len(test_data))
    word2id, id2word, max_plot_len, max_ending_len, min_ending_len = creat_vocab(train_data,None)
    word_vocab_size = len(word2id.keys())
    print('vocab size:', word_vocab_size)



