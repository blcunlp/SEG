
import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
from six.moves import xrange
from model import SCST_RLModel
import util
import pickle
from tensorflow.python import debug as tf_debug
from data_utils import prepare_dataset, creat_vocab, get_batches, \
  prepare_data_for_beam_seach_decode,result_ids2words,\
  compute_bleu_reward, compute_rouge_or_cider_reward, prepare_batch_data,mask_base_samp_tatget_sequences
import beam_search

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', './data/ROC_data', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'rl_train', 'must be one of seq2seq_train/rl_train/beam_search_decode/greedy_decode')
tf.app.flags.DEFINE_string('reward_type', 'bleu_1', 'which type to choose as reward, i.e. bleu_1, bleu_2, bleu_3, bleu_4, rouge, cider')

# Where to save output
tf.app.flags.DEFINE_string('exp_name', '0201', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
tf.app.flags.DEFINE_string('decode_ckpt_dir', 'eval_seq2seq', 'directory of model saved to decode.')

# Hyperparameters
tf.app.flags.DEFINE_integer('epochs', 50, "Maximum of epochs when training")
tf.app.flags.DEFINE_integer('eval_global_step', 100, "starting eval per eval_global_step")
tf.app.flags.DEFINE_integer('save_model_secs',120, "seconds for saving models")
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 512, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 64, 'minibatch size')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 2, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('word_vocab_size', None, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to None, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('max_grad_norm', 5.0, 'for gradient clipping')
tf.app.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')
tf.app.flags.DEFINE_float('rl_loss_scale_factor', 1, 'rate of rl loss')

# Pointer-generator or baseline ——
tf.app.flags.DEFINE_boolean('pointer_gen', False, 'If True, use pointer-generator model. If False, use baseline model.')

# For mixed loss
tf.app.flags.DEFINE_boolean('use_mixed_loss', False, 'If True, use mixed_loss.')
tf.app.flags.DEFINE_float('loss_rate_of_sem', 1, 'rate of semantic relevence loss')
tf.app.flags.DEFINE_float('loss_rate_of_mle', 1, 'rate of cross entory loss')
# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', False, "Run in tensorflow's debug mode (watches for NaN/inf values)")

def convert_to_coverage_model():
  """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
  tf.logging.info("converting non-coverage model to coverage model..")

  # initialize an entire coverage model from scratch
  sess = tf.Session(config=util.get_config())
  print ("initializing everything...")
  sess.run(tf.global_variables_initializer())

  # load all non-coverage weights from checkpoint
  saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adam" not in v.name])
  print ("restoring non-coverage variables...")
  curr_ckpt = util.load_ckpt(saver, sess, ckpt_dir="train_seq2seq")
  print ("restored.")

  # save this model and quit
  new_fname = curr_ckpt + '_cov_init'
  print ("saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver() # this one will save all variables that now exist
  new_saver.save(sess, new_fname)
  print ("saved.")
  exit()

def run_seq2seq_training(model_train, model_eval, train_data, train_batches, valid_data, valid_batches, word2id, max_ending_len, sv, sess_context_manager):
  """Repeatedly runs training iterations, logging loss to screen"""
  tf.logging.info("starting run seq2seq_training")
  with sess_context_manager as sess:
    if FLAGS.debug: # start the tensorflow debugger
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    ones_reward = np.ones([FLAGS.batch_size])

    best_eval_loss = 100.0  # will hold the best loss achieved so far
    for ep in xrange(FLAGS.epochs):
      sys.stdout.flush()
      print('<Epoch {}>'.format(ep))
      np.random.shuffle(train_data)
      np.random.shuffle(valid_data)
      train_loss, train_coverage_loss = 0.0, 0.0
      start_time = time.time()
      i = 0
      for start, end in train_batches:
        i += 1
        batch_train_data = prepare_batch_data(train_data, start, end, word2id, max_ending_len, FLAGS.pointer_gen)
        batch_train_plot, batch_train_ending, batch_train_ext_plot, batch_train_len_plot, batch_train_len_ending, train_max_plot_oovs,train_plot_mask = batch_train_data
        train_results = model_train.run_train_step(sess,
                                       batch_train_plot,
                                       batch_train_ending,
                                       batch_train_ext_plot,
                                       batch_train_len_plot,
                                       batch_train_len_ending,
                                       train_max_plot_oovs,
                                       train_plot_mask,
                                       ones_reward
                                       )
        batch_train_loss = train_results['loss']
        train_loss += batch_train_loss
        train_step = train_results['global_step']
        if FLAGS.coverage:
          batch_train_coverage_loss = train_results['coverage_loss']
          train_coverage_loss += batch_train_coverage_loss

        if train_step % FLAGS.eval_global_step == 0:
          sys.stdout.flush()
          print('-'*50)
          print('train_step = {0:d}, step_train_loss = {1:.6f}'.format(train_step,batch_train_loss))
          print('Starting Dev for early_stopping')
          eval_dir = os.path.join(FLAGS.exp_name, "eval_seq2seq")  # make a subdir of the root dir for eval data
          bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')  # this is where checkpoints of best models are saved
          eval_loss, eval_coverage_loss = 0, 0
          count = 0
          for start, end in valid_batches:
            count += 1
            batch_valid_data = prepare_batch_data(valid_data, start, end, word2id, max_ending_len, FLAGS.pointer_gen)
            batch_valid_plot, batch_valid_ending, batch_valid_ext_plot, batch_valid_len_plot, batch_valid_len_ending, valid_max_plot_oovs,vaild_plot_mask = batch_valid_data
          # run eval on the batch
            eval_results = model_eval.run_eval_step(sess,
                                        batch_valid_plot,
                                        batch_valid_ending,
                                        batch_valid_ext_plot,
                                        batch_valid_len_plot,
                                        batch_valid_len_ending,
                                        valid_max_plot_oovs,
                                        vaild_plot_mask,
                                        ones_reward)

            batch_eval_loss = eval_results['loss']
            eval_loss += batch_eval_loss
            if FLAGS.coverage:
              batch_coverage_loss = eval_results['coverage_loss']
              eval_coverage_loss += batch_coverage_loss
          eval_loss = eval_loss / count
          # If eval_loss is best so far, save this checkpoint (early stopping).
          print('best_eval_loss: {0:.6f}, eval_loss: {1:.6f}'.format(best_eval_loss, eval_loss, batch_train_loss))
          if eval_loss < best_eval_loss:
            sv.saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            sys.stdout.flush()
            tf.logging.info('Found new best model with %.3f eval_loss. Saved to %s', eval_loss, bestmodel_save_path)
            best_eval_loss = eval_loss

      end_time = time.time()
      print('time on epoch {0:d}={1:.6f}'.format(ep, end_time - start_time))
      tf.logging.info('loss: %6f', train_loss / i) # print the loss to screen
      if  FLAGS.coverage:
        tf.logging.info("coverage_loss: %6f", train_coverage_loss / i)  # print the coverage loss to screen

      if not np.isfinite(train_loss):
        raise Exception("Loss is not finite. Stopping.")
      sys.stdout.flush()

def run_rl_training(model_train, model_eval, train_data, train_batches, valid_data, valid_batches, word2id,max_ending_len, sv, sess_context_manager):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run rl_training")
  with sess_context_manager as sess:
    if FLAGS.debug: # start the tensorflow debugger
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    ones_reward = np.ones([FLAGS.batch_size])
    best_eval_loss = 10000.0  # will hold the best loss achieved so far
    for ep in xrange(FLAGS.epochs):
      sys.stdout.flush()
      print('<Epoch {}>'.format(ep))
      np.random.shuffle(train_data)
      np.random.shuffle(valid_data)
      train_loss, train_rl_loss, train_coverage_loss = 0.0, 0.0, 0.0
      start_time = time.time()
      i = 0
      for start, end in train_batches:
        i += 1
        batch_train_data = prepare_batch_data(train_data, start, end, word2id,max_ending_len, FLAGS.pointer_gen)
        batch_train_plot, batch_train_ending, batch_train_ext_plot, batch_train_len_plot, batch_train_len_ending, train_max_plot_oovs, train_plot_mask = batch_train_data

        seq2seq_train_results = model_eval.run_rl_step(sess,
                                       batch_train_plot,
                                       batch_train_ending,
                                       batch_train_ext_plot,
                                       batch_train_len_plot,
                                       batch_train_len_ending,
                                       train_max_plot_oovs,
                                       train_plot_mask,
                                       ones_reward)
        seq2seq_step = seq2seq_train_results['global_step']
        train_sample_sequences = seq2seq_train_results['sample_seqs']
        train_base_sequences = seq2seq_train_results['base_seqs']
        train_sample_dists = seq2seq_train_results['sample_dists_rl_step']
        # compute r_sample-r_base as reward
        masked_train_base_sequences, masked_train_sample_sequences, masked_train_ending_sequences = mask_base_samp_tatget_sequences(
          train_base_sequences, train_sample_sequences, batch_train_ending, batch_train_len_ending)
        # compute r_sample-r_base as reward
        if FLAGS.reward_type.split('_')[0] == 'bleu':
          train_r = compute_bleu_reward(masked_train_base_sequences, masked_train_sample_sequences,
                                        masked_train_ending_sequences, n_gram=int(FLAGS.reward_type.split('_')[1]))
        elif FLAGS.reward_type in ['rouge', 'cider']:
          if FLAGS.reward_type == 'cider':
            train_cached_tokens = './data/ROC_data/ROC_train_ending-idxs.p'
            train_pkl_file = pickle.load(open(train_cached_tokens, 'rb'))
          else:
            train_pkl_file = None
          train_r = compute_rouge_or_cider_reward(masked_train_base_sequences, masked_train_sample_sequences,
                                        masked_train_ending_sequences, metric=FLAGS.reward_type,pkl_file = train_pkl_file)
        else:
          print( 'Error: reward_type must be one of bleu1-4, rouge_l and cider' )
          train_r = None
          exit()
        train_results = model_train.run_train_step(sess,
                                         batch_train_plot,
                                         batch_train_ending,
                                         batch_train_ext_plot,
                                         batch_train_len_plot,
                                         batch_train_len_ending,
                                         train_max_plot_oovs,
                                         train_plot_mask,
                                         train_r,
                                         train_sample_dists,
                                         )
        batch_train_loss = train_results['loss']
        batch_train_rl_loss = train_results['rl_loss']
        train_loss += batch_train_loss
        train_rl_loss += batch_train_rl_loss
        train_step = train_results['global_step']
        if FLAGS.coverage:
          batch_train_coverage_loss = train_results['coverage_loss']
          train_coverage_loss += batch_train_coverage_loss

        if train_step % FLAGS.eval_global_step == 0:
          sys.stdout.flush()
          print('-'*50)
          print('Starting Dev for early_stopping')
          print('seq2seq_step = {0:d}, train_step = {1:d}, step_train_loss = {2:.6f}, step_train_rl_loss = {3:.6f}'.format(seq2seq_step,train_step,batch_train_loss,batch_train_rl_loss))
          eval_dir = os.path.join(FLAGS.exp_name, "eval_rl" + '_' + FLAGS.reward_type+ 'mu_' + str(FLAGS.rl_loss_scale_factor))  # make a subdir of the root dir for eval data
          bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')  # this is where checkpoints of best models are saved
          eval_loss, eval_rl_loss, eval_coverage_loss = 0, 0, 0
          count = 0
          for start, end in valid_batches:
            count += 1
            batch_valid_data = prepare_batch_data(valid_data, start, end, word2id,max_ending_len, FLAGS.pointer_gen)
            batch_valid_plot, batch_valid_ending, batch_valid_ext_plot, batch_valid_len_plot, batch_valid_len_ending, valid_max_plot_oovs, vaild_plot_mask = batch_valid_data
          # run eval on the batch
            seq2seq_eval_results = model_eval.run_rl_step(sess,
                                        batch_valid_plot,
                                        batch_valid_ending,
                                        batch_valid_ext_plot,
                                        batch_valid_len_plot,
                                        batch_valid_len_ending,
                                        valid_max_plot_oovs,
                                        vaild_plot_mask,
                                        ones_reward)
            eval_sample_sequences = seq2seq_eval_results['sample_seqs']
            eval_base_sequences = seq2seq_eval_results['base_seqs']
            eval_sample_dists = seq2seq_eval_results['sample_dists_rl_step']

            masked_eval_base_sequences, masked_eval_sample_sequences, masked_eval_ending_sequences = mask_base_samp_tatget_sequences(
              eval_base_sequences, eval_sample_sequences, batch_valid_ending, batch_valid_len_ending)
            # compute r_sample-r_base as reward
            if FLAGS.reward_type.split('_')[0] == 'bleu':
              eval_r = compute_bleu_reward(masked_eval_base_sequences, masked_eval_sample_sequences,
                                           masked_eval_ending_sequences, n_gram=int(FLAGS.reward_type.split('_')[1]))
            elif FLAGS.reward_type in ['rouge', 'cider']:
              if FLAGS.reward_type == 'cider':
                eval_cached_tokens = './data/ROC_data/ROC_eval_ending-idxs.p'
                eval_pkl_file = pickle.load(open(eval_cached_tokens, 'rb'))
              else:
                eval_pkl_file = None
              eval_r = compute_rouge_or_cider_reward(masked_eval_base_sequences, masked_eval_sample_sequences,
                                                     masked_eval_ending_sequences, metric=FLAGS.reward_type, pkl_file = eval_pkl_file)
            else:
              print('Error: reward_type must be one of bleu1-4, rouge_l and cider')
              eval_r = None
              exit()
            eval_results = model_eval.run_eval_step(sess,
                                                    batch_valid_plot,
                                                    batch_valid_ending,
                                                    batch_valid_ext_plot,
                                                    batch_valid_len_plot,
                                                    batch_valid_len_ending,
                                                    valid_max_plot_oovs,
                                                    vaild_plot_mask,
                                                    eval_r,
                                                    eval_sample_dists)
            batch_eval_loss = eval_results['loss']
            batch_eval_rl_loss = eval_results['rl_loss']
            eval_loss += batch_eval_loss
            eval_rl_loss += batch_eval_rl_loss
            if FLAGS.coverage:
              batch_coverage_loss = eval_results['coverage_loss']
              eval_coverage_loss += batch_coverage_loss
          eval_loss = eval_loss / count
          eval_rl_loss = eval_rl_loss / count
          print('best_eval_loss: {0:.6f}, eval_loss: {1:.6f}, eval_rl_loss: {2:.6f}'.format(best_eval_loss, eval_loss, eval_rl_loss))
          # If running_avg_loss is best so far, save this checkpoint (early stopping).
          # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
          if eval_loss < best_eval_loss:
            sys.stdout.flush()
            tf.logging.info('Found new best model with %.3f eval_loss. Saving to %s', eval_loss, bestmodel_save_path)
            sv.saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
            best_eval_loss = eval_loss

      end_time = time.time()
      print('time on epoch {0:d}={1:.6f}'.format(ep, end_time - start_time))
      tf.logging.info('avg_epoch_loss: %6f', train_loss / i) # print the loss to screen
      tf.logging.info('avg_epoch_rl_loss: %6f', train_rl_loss / i) # print the loss to screen
      if  FLAGS.coverage:
        tf.logging.info("coverage_loss: %6f", train_coverage_loss / i)  # print the coverage loss to screen

      if not np.isfinite(train_loss):
        raise Exception("Loss is not finite. Stopping.")
      sys.stdout.flush()

def run_beam_search_decode(model, examples_list, id2word, data, ckpt_dir):
  ones_reward = np.ones([FLAGS.batch_size])
  saver = tf.train.Saver()  # we use this to load checkpoints for decoding
  sess = tf.Session(config=util.get_config())
  # Load an initial checkpoint to use for decoding
  ckpt_path = util.load_ckpt(saver, sess, ckpt_dir=ckpt_dir)
  print('Finished loading model')
  ckpt_name = 'BS_'+ data + "-ckpt_from-" + ckpt_dir
  ckpt_name = ckpt_name + ckpt_path.split('-')[-1] if ckpt_dir.split('_')[0] == 'eval'else ckpt_name # this is something of the form "ckpt-123456"
  decode_dir = os.path.join(FLAGS.exp_name, ckpt_name)
  if os.path.exists(decode_dir):
    raise Exception("single_pass decode directory %s should not already exist" % decode_dir)
  if not os.path.exists(decode_dir): os.mkdir(decode_dir)
  test_file_name = '{}/{}_results.txt'.format(decode_dir, data)
  f = open(test_file_name, 'w')
  for i in xrange(len(examples_list)):
    if i%200 == 0:
      print(i)
    batch_test_plot, batch_test_ending, batch_test_ext_plot, batch_test_len_plot, batch_test_len_ending, test_max_plot_oovs, batch_plot_oovs, batch_plot_mask = examples_list[i]
    best_hyp = beam_search.run_beam_search(sess, model, id2word, batch_test_plot, batch_test_ending, batch_test_ext_plot,batch_test_len_plot, batch_test_len_ending, test_max_plot_oovs, batch_plot_mask, ones_reward)

    # Extract the output ids from the hypothesis and convert back to words
    output_ids = [int(t) for t in best_hyp.tokens[1:]]
    logit_words = result_ids2words(output_ids, id2word, batch_plot_oovs[0])
    if 'PAD' not in logit_words:
      logit_words.append('PAD')
    mask_logit = logit_words[0:logit_words.index('PAD')]
    logit_txt = ' '.join(mask_logit)
    new_line = '{}\n'.format(logit_txt)
    f.write(new_line)
  print('Finished writing results!')

def main(unused_argv):
  if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
    raise Exception("Problem with flags: %s" % unused_argv)

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  if FLAGS.mode == 'rl_train':
    tf.logging.info('Starting model in %s mode...', FLAGS.mode + '_' + FLAGS.reward_type)
  else:
    tf.logging.info('Starting model in %s mode...', FLAGS.mode)
  # If in decode mode, set batch_size = beam_size
  # Reason: in decode mode, we decode one example at a time.
  # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
  if FLAGS.mode == 'beam_search_decode':
    FLAGS.batch_size = FLAGS.beam_size

  train_data, valid_data, test_data = prepare_dataset(FLAGS.data_path)
  print('TrainData Size:', len(train_data))
  print('ValidData Size:', len(valid_data))
  print('TestData Size:', len(test_data))

  print("Building vocabulary ..... ")
  word2id, id2word, _, max_ending_len, min_ending_len = creat_vocab(train_data, FLAGS.word_vocab_size)
  print("Finished building vocabulary!")
  word_vocab_size = len(word2id.keys())

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['mode', 'loss_rate_of_sem','loss_rate_of_mle','word_vocab_size', 'use_mixed_loss', 'lr', 'train_keep_prob','rl_loss_scale_factor', 'rand_unif_init_mag', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'coverage', 'cov_loss_wt', 'pointer_gen']
  hps_dict = {}
  for key,val in FLAGS.__flags.items(): # for each flag
    if key in hparam_list: # if it's in the list
      hps_dict[key] = val # add it to the dict
  hps_dict['max_dec_steps'] = max_ending_len
  hps_dict['min_ending_len'] = min_ending_len
  if FLAGS.word_vocab_size == None:
    hps_dict['word_vocab_size'] = word_vocab_size
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  # create minibatches of data
  train_batches = get_batches(len(train_data), FLAGS.batch_size)
  valid_batches = get_batches(len(valid_data), FLAGS.batch_size)

  tf.set_random_seed(111) # a seed value for randomness

  if hps.mode == 'seq2seq_train':
    train_dir = os.path.join(FLAGS.exp_name, "train_seq2seq")
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    with tf.Graph().as_default():
      initializer = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag)
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
          m_train = SCST_RLModel(is_training=True, hps=hps)
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
          m_valid = SCST_RLModel(is_training=False, hps=hps)
      if FLAGS.convert_to_coverage_model:
        assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model()
      sv = tf.train.Supervisor(logdir=train_dir,save_model_secs=FLAGS.save_model_secs)

      sess_context_manager = sv.managed_session(config=util.get_config())
      tf.logging.info("Created session.")
      try:
        run_seq2seq_training(m_train, m_valid, train_data, train_batches, valid_data, valid_batches, word2id,max_ending_len,
                     sv, sess_context_manager)  # this is an infinite loop until interrupted
      except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()

  elif hps.mode == 'rl_train':
    train_dir = os.path.join(FLAGS.exp_name, "train_rl" + '_' + FLAGS.reward_type + 'mu_' + str(FLAGS.rl_loss_scale_factor))
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    with tf.Graph().as_default():
      initializer = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag)
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m_train = SCST_RLModel(is_training=True, hps=hps)
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        m_valid = SCST_RLModel(is_training=False, hps=hps)

      # define load_pretrain funtion for restoring best seq2seq model from eval_dir
      ckpt_dir = 'eval_seq2seq'
      latest_filename = "checkpoint_best" if ckpt_dir == "eval_seq2seq" else None
      ckpt_dir = os.path.join(FLAGS.exp_name, ckpt_dir)
      ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
      print("loading pre_trained seq2seq model from %s",ckpt_state.model_checkpoint_path)
      saver = tf.train.Saver()
      def load_pretrain(sess):
        return saver.restore(sess, ckpt_state.model_checkpoint_path)

      sv = tf.train.Supervisor(logdir=train_dir,
                               saver = saver,
                               save_model_secs=FLAGS.save_model_secs,
                               init_fn=load_pretrain)
      sess_context_manager = sv.managed_session(config=util.get_config())
      tf.logging.info("Created session.")
      try:
        run_rl_training(m_train, m_valid, train_data, train_batches, valid_data, valid_batches, word2id, max_ending_len,
                     sv, sess_context_manager)  # this is an infinite loop until interrupted
      except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()

  elif hps.mode == 'beam_search_decode':
    # This will be the hyperparameters for the decoder model
    decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
    test_examples_list = prepare_data_for_beam_seach_decode(test_data, FLAGS.batch_size, word2id, max_plot_len, max_ending_len,FLAGS.pointer_gen)
    with tf.Graph().as_default():
      initializer = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag)
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
          model_test = SCST_RLModel(is_training=False, hps=decode_model_hps)
          run_beam_search_decode(model_test, test_examples_list, id2word, data = 'test_data', ckpt_dir =FLAGS.decode_ckpt_dir)
  else:
    raise ValueError("The 'mode' flag must be one of seq2seq_train/rl_train/beam_search_decode")

if __name__ == '__main__':
  tf.app.run()
