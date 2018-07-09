
"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, BasicLSTMCell,MultiRNNCell
from attention_decoder import attention_decoder, attention_generator
from tensorflow.contrib.rnn import DropoutWrapper, ResidualWrapper
import data_utils
from six.moves import xrange
from loss import sequence_loss,rl_loss


class SCST_RLModel(object):
  """A class to represent a sequence-to-sequence or RL model for story ending genaration."""

  def __init__(self, is_training, hps):
    self._hps = hps
    self._is_training = is_training
    for key, value in sorted(self._hps._asdict().items()):
      print(str(key) + ':' + str(value))
    print('--'*30)
    self.build_graph()

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps

    # encoder part
    self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size,], name='enc_lens')
    self.enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, None], name='enc_padding_mask')
    if hps.pointer_gen:
      self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab')
      self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

    # decoder part
    self.decoder_inputs = tf.placeholder(tf.int32, shape=(hps.batch_size, hps.max_dec_steps), name='decoder_inputs')
    self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(hps.batch_size,), name='decoder_inputs_length')

    decoder_start_token = tf.ones(shape=[hps.batch_size, 1], dtype=tf.int32) * data_utils.BOS_ID
    decoder_end_token = tf.ones(shape=[hps.batch_size, 1], dtype=tf.int32) * data_utils.PAD_ID

    self._dec_batch = tf.concat([decoder_start_token, self.decoder_inputs], axis=1)
    self._dec_lens = self.decoder_inputs_length + 1
    self._target_batch = tf.concat([self.decoder_inputs, decoder_end_token], axis=1)

    if self._hps.mode in ["beam_search_decode", "greedy_decode"] and hps.coverage:
      self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size, None], name='prev_coverage')
    self.reward = tf.placeholder(tf.float32, [hps.batch_size,])
    if hps.mode == 'rl_train':
      self.sample_dists = tf.placeholder(tf.float32, shape=(hps.batch_size, None),name = 'sample_dists')

  def _make_feed_dict(self, plot, ending, ext_plot, len_plot, len_ending, max_plot_oovs,plot_mask, reward, sample_dists, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._enc_batch] = plot
    feed_dict[self._enc_lens] = len_plot
    feed_dict[self.enc_padding_mask] = plot_mask
    if self._hps.pointer_gen:
      feed_dict[self._enc_batch_extend_vocab] = ext_plot
      feed_dict[self._max_art_oovs] = max_plot_oovs
    if not just_enc:
      feed_dict[self.decoder_inputs] = ending
      feed_dict[self.decoder_inputs_length] = len_ending
    feed_dict[self.reward] = reward
    if sample_dists is not None:
      feed_dict[self.sample_dists] = sample_dists
    return feed_dict

  def _add_encoder(self, encoder_inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope('encoder'):
      cell_fw = LSTMCell(self._hps.hidden_dim, state_is_tuple=True)
      cell_bw = LSTMCell(self._hps.hidden_dim, state_is_tuple=True)
      if self._is_training:
        cell_fw = DropoutWrapper(cell_fw, input_keep_prob=1 - self._hps.train_keep_prob,
                                 output_keep_prob=1 - self._hps.train_keep_prob)
        cell_bw = DropoutWrapper(cell_bw, input_keep_prob=1 - self._hps.train_keep_prob,
                                 output_keep_prob=1 - self._hps.train_keep_prob)

      (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states
    return encoder_outputs, fw_st, bw_st

  def _reduce_states(self, fw_st, bw_st):
    """Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

    Args:
      fw_st: LSTMStateTuple with hidden_dim units.
      bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
      state: LSTMStateTuple with hidden_dim units.
    """
    hidden_dim = self._hps.hidden_dim
    with tf.variable_scope('reduce_final_st'):
      # Define weights and biases to reduce the cell and reduce the state
      w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32)
      w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32)
      bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32)
      bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32)

      # Apply linear layer
      old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
      old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
      new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
      new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
      return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

  def _calc_final_dist(self, vocab_dists, attn_dists):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      vocab_dists = [p_gen * dist for (p_gen,dist) in zip(self.p_gens, vocab_dists)]
      attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(self.p_gens, attn_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extended_vsize = self._hps.word_vocab_size + self._max_art_oovs # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
      vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      batch_nums = tf.range(0, limit=self._hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
      indices = tf.stack( (batch_nums, self._enc_batch_extend_vocab), axis=2) # shape (batch_size, enc_t, 2)
      shape = [self._hps.batch_size, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

      return final_dists

  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps

    with tf.variable_scope('seq2seq'):
      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        self.embedding = tf.get_variable('embedding', [hps.word_vocab_size, hps.emb_dim], dtype=tf.float32)
        emb_enc_inputs = tf.nn.embedding_lookup(self.embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)

        if hps.mode in ['seq2seq_train', 'rl_train']:
          emb_dec_inputs = [tf.nn.embedding_lookup(self.embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)
        else:
          emb_dec_inputs = [tf.nn.embedding_lookup(self.embedding, x) for x in tf.unstack(self.decoder_inputs, axis=1)]  # list length max_dec_steps containing shape (batch_size, emb_size)

        if self._is_training and hps.train_keep_prob < 1:
            emb_enc_inputs = tf.nn.dropout(emb_enc_inputs, hps.train_keep_prob)
            emb_dec_inputs = [tf.nn.dropout(emb_dec_input, hps.train_keep_prob) for emb_dec_input in emb_dec_inputs]
      # Add the encoder.
      with tf.variable_scope('encoder'):
        self._enc_states, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens)
      # Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
        self._dec_in_state = self._reduce_states(fw_st, bw_st)

      # Add the decoder.
      with tf.variable_scope('decoder'):
        self.deco_cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim, state_is_tuple=True)
        if self._is_training:
          cell = DropoutWrapper(self.deco_cell, input_keep_prob=1 - self._hps.train_keep_prob,
                                output_keep_prob=1 - self._hps.train_keep_prob)
        else:
          cell = self.deco_cell
        prev_coverage = self.prev_coverage if hps.mode == "beam_search_decode" and hps.coverage else None  # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time
        if hps.mode == 'rl_train':
          self.base_seqs, self.sample_seqs, self.sample_dists_rl_step, _, _, _, _, _, _ = attention_generator(self.embedding,
                                                                                                              self._hps.max_dec_steps+1,
                                                                                                              self._dec_in_state,
                                                                                                              self._enc_states,
                                                                                                              self.enc_padding_mask,
                                                                                                              self.deco_cell,
                                                                                                              initial_state_attention=True,
                                                                                                              pointer_gen=hps.pointer_gen,
                                                                                                              use_coverage=hps.coverage,
                                                                                                              prev_coverage=np.zeros([hps.batch_size, self._hps.max_dec_steps]))

        decoder_outputs, vocab_scores, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = attention_decoder(hps.mode,
                                                                                                                        emb_dec_inputs,
                                                                                                                        hps.word_vocab_size,
                                                                                                                        self._dec_in_state,
                                                                                                                        self._enc_states,
                                                                                                                        self.enc_padding_mask,
                                                                                                                        cell,
                                                                                                                        initial_state_attention=(hps.mode=="beam_search_decode"),
                                                                                                                        pointer_gen=hps.pointer_gen,
                                                                                                                        use_coverage=hps.coverage,
                                                                                                                        prev_coverage=prev_coverage
                                                                                                                        )
        self.vocab_dists = [tf.nn.softmax(s) for s in vocab_scores] # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

      # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
        if hps.pointer_gen:
          final_dists = self._calc_final_dist(self.vocab_dists, self.attn_dists)
        else: # final distribution is just vocabulary distribution
          final_dists = self.vocab_dists

        if hps.mode in ['seq2seq_train', 'rl_train'] :
          with tf.variable_scope('loss'):
            dec_padding_mask = tf.sequence_mask(lengths=self._dec_lens, maxlen=hps.max_dec_steps + 1,dtype=tf.float32, name='dec_masks')

            if hps.pointer_gen:
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
              loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
              batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
              for dec_step, dist in enumerate(final_dists):
                targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
                indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
                gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
                losses = -tf.log(gold_probs)
                loss_per_step.append(losses)
            # Apply dec_padding_mask and get loss
              self._loss = _mask_and_avg(loss_per_step, dec_padding_mask) # mle loss for pointer-gen

            else: # baseline model
              self._loss = sequence_loss(tf.stack(vocab_scores, axis=1), self._target_batch, dec_padding_mask) # mle loss for seq2seq_attn

            if hps.mode == 'rl_train':
              self.prob_sample, self.masked_sample_dists,self._loss_rl = rl_loss(self.sample_dists, dec_padding_mask, self.reward) # compute rl loss
              self._loss = hps.rl_loss_scale_factor * self._loss_rl + (1 - hps.rl_loss_scale_factor) * self._loss # W1 * rl_loss + (1 - W1)* mle_loss

            if hps.use_mixed_loss: # compute mixed loss
            # compute semantic relevance loss
              self.v_s = self.enco_final_h  # _dec_out_state
              self.dec_out_state = self._dec_out_state[1]
              self.v_t = tf.subtract(self.dec_out_state, self.v_s)
              v_s_t = tf.reduce_sum(tf.multiply(self.v_s, self.v_t), axis=1)
              v_s_s = tf.sqrt(tf.reduce_sum(tf.multiply(self.v_s, self.v_s), axis=1))
              v_t_t = tf.sqrt(tf.reduce_sum(tf.multiply(self.v_t, self.v_t), axis=1))
              cos_v_s_t = tf.div(v_s_t, tf.multiply(v_s_s, v_t_t))
              self.final_semantic_loss = tf.reduce_mean(cos_v_s_t)
              # self.decoder_pred_train = tf.argmax(tf.stack(vocab_scores, axis=1), axis=-1, name='decoder_pred_train')

              self._loss = hps.loss_rate_of_mle*self._loss - hps.loss_rate_of_sem*self.final_semantic_loss
            if hps.coverage:
              with tf.variable_scope('coverage_loss'):
                self._coverage_loss = _coverage_loss(self.attn_dists, dec_padding_mask)
              self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
        if hps.mode == "beam_search_decode":
      # We run decode beam search mode one decoder step at a time
          assert len(final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
          final_dists = final_dists[0]
          topk_probs, self._topk_ids = tf.nn.top_k(final_dists, hps.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
          self._topk_log_probs = tf.log(topk_probs)

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    self.tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, self.tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
    # Clip the gradients
    with tf.device("/gpu:0"):
      grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
    # Apply adam optimizer
    optimizer = tf.train.AdamOptimizer(self._hps.lr)
    with tf.device("/gpu:0"):
      self._train_op = optimizer.apply_gradients(zip(grads, self.tvars), global_step=self.global_step, name='train_step')

  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    self._add_placeholders()
    with tf.device("/gpu:0"):
      self._add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._is_training:
      self._add_train_op()

  def run_train_step(self, sess, plot, ending, ext_plot, len_plot, len_ending, max_plot_oovs, plot_mask, reward, sample_dists = None):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(plot, ending, ext_plot, len_plot,
                                     len_ending, max_plot_oovs, plot_mask, reward, sample_dists, just_enc=False)
    to_return = {
        'train_op': self._train_op,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    if self._hps.mode == 'rl_train':
      to_return['rl_loss'] = self._loss_rl

    return sess.run(to_return, feed_dict)

  def run_rl_step(self, sess, plot, ending, ext_plot, len_plot, len_ending, max_plot_oovs, plot_mask,reward):
    feed_dict = self._make_feed_dict(plot, ending, ext_plot, len_plot,
                                     len_ending, max_plot_oovs, plot_mask, reward, sample_dists= None,just_enc=True)
    to_return = {
      'base_seqs': self.base_seqs,
      'sample_seqs': self.sample_seqs,
      'sample_dists_rl_step': self.sample_dists_rl_step,
      'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, plot, ending, ext_plot, len_plot, len_ending, max_plot_oovs, plot_mask, reward, sample_dists=None):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(plot, ending, ext_plot, len_plot,
                                     len_ending, max_plot_oovs, plot_mask, reward, sample_dists, just_enc=False)
    to_return = {
        'loss': self._loss,
        'global_step': self.global_step,
    }
    if self._hps.coverage:
      to_return['coverage_loss'] = self._coverage_loss
    if self._hps.mode == 'rl_train':
      to_return['rl_loss'] = self._loss_rl
    return sess.run(to_return, feed_dict)

  def run_encoder(self, sess, plot, ending, ext_plot, len_plot, len_ending, max_plot_oovs, plot_mask, reward):
    """For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

    Args:
      sess: Tensorflow session.
      batch: Batch object that is the same example repeated across the batch (for beam search)

    Returns:
      enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
      dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
    """
    feed_dict = self._make_feed_dict(plot, ending,ext_plot, len_plot, len_ending, max_plot_oovs, plot_mask, reward, sample_dists = None, just_enc=True) # feed the batch into the placeholders
    (enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step], feed_dict) # run the encoder
    # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
    dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
    return enc_states, dec_in_state

  def decode_onestep(self, sess, plot_mask, ext_plot, max_plot_oovs,enc_len, latest_tokens, enc_states, dec_init_states, prev_coverage):
    """For beam search decoding. Run the decoder for one step."""
    beam_size = len(dec_init_states)

    # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
    cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
    hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
    new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
    new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
    new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

    feed = {
        self._enc_states: enc_states,
        self._enc_lens: enc_len,
        self._dec_in_state: new_dec_in_state,
        self.decoder_inputs: np.transpose(np.array([latest_tokens])),
        self.enc_padding_mask: plot_mask
    }

    to_return = {
      "ids": self._topk_ids,
      "probs": self._topk_log_probs,
      "states": self._dec_out_state,
      "attn_dists": self.attn_dists
    }

    if self._hps.pointer_gen:
      feed[self._enc_batch_extend_vocab] = ext_plot
      feed[self._max_art_oovs] = max_plot_oovs
      to_return['p_gens'] = self.p_gens

    if self._hps.coverage:
      feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
      to_return['coverage'] = self.coverage

    results = sess.run(to_return, feed_dict=feed) # run the decoder step

    # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
    new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in xrange(beam_size)]

    # Convert singleton list containing a tensor to a list of k arrays
    assert len(results['attn_dists'])==1
    attn_dists = results['attn_dists'][0].tolist()

    if self._hps.pointer_gen:
      # Convert singleton list containing a tensor to a list of k arrays
      assert len(results['p_gens'])==1
      p_gens = results['p_gens'][0].tolist()
    else:
      p_gens = [None for _ in xrange(beam_size)]

    # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
    if self._hps.coverage:
      new_coverage = results['coverage'].tolist()
      assert len(new_coverage) == beam_size
    else:
      new_coverage = [None for _ in xrange(beam_size)]

    return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage

def _mask_and_avg(values, padding_mask):
  """Applies mask to values then returns overall average (a scalar)

  Args:
    values: a list length max_dec_steps containing arrays shape (batch_size).
    padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

  Returns:
    a scalar
  """

  dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
  values_per_step = [v * padding_mask[:,dec_step] for dec_step,v in enumerate(values)]
  values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
  return tf.reduce_mean(values_per_ex) # overall average

def _coverage_loss(attn_dists, padding_mask):
  """Calculates the coverage loss from the attention distributions.

  Args:
    attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
    padding_mask: shape (batch_size, max_dec_steps).

  Returns:
    coverage_loss: scalar
  """
  coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
  covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
  for a in attn_dists:
    covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
    covlosses.append(covloss)
    coverage += a # update the coverage vector
  coverage_loss = _mask_and_avg(covlosses, padding_mask)
  return coverage_loss



