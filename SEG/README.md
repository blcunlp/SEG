This repository contains code for the NLPCC 2018 paper *[From Plots to Endings: A Reinforced Pointer Generator for Story Ending Generation](https://arxiv.org/abs/1704.04368)*. 

## Description
Our code is based on the [pointer-generator](https://github.com/abisee/pointer-generator) for the ACL 2017 paper [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368).
We build three modes in the code: *seq2seq_train, rl_train, and beam_search_decode*. 

## Requirements
 - python 3.5
 - tensorflow 1.4.0
 - numpy 1.14.4 
 - nlgeval

We use the evaluation package [nlg-eval](https://github.com/Maluuba/nlg-eval) for automatic evaluation. For more detailed information, see the following paper: [Relevance of Unsupervised Metrics in Task-Oriented Dialogue for Evaluating Natural Language Generation](http://arxiv.org/abs/1706.09799).

## How to run

### Pretrain Seq2seq/Pointer-generator+coverage Model
A directory will be created according to the super-parameter `exp_name`. Two subdirectory `train_seq2seq` and `eval_seq2seq` of `exp_name` will be produced to save the checkpoints during training and validation.

- run seq2seq model 
```
python3 run.py --mode seq2seq_train --data_path ./data/ROC_data --exp_name model_seq2seq --pointer_gen False --word_vocab_size None --coverage False --convert_to_coverage_model False >logs/seq2seq
```
- run pointer-generator+coverage model
1. train pointer-generator for 10 epochs
 ```
python3 run.py --mode seq2seq_train --data_path ./data/ROC_data --exp_name model_pgen10e --epochs 10 --pointer_gen True --word_vocab_size 15000 --coverage False --convert_to_coverage_model False >logs/pgen10e
 ```
2. convert pointer-generator to a model for running coverage
 ```
python3 run.py --mode seq2seq_train --data_path ./data/ROC_data --exp_name model_pgen10e --pointer_gen True --word_vocab_size 15000 --coverage True --convert_to_coverage_model True
 ```
3. continue to train pointer-generator model with coverage
 ```
python3 run.py --exp_name model_pgen10e_10cov --mode seq2seq_train --epochs 10 --pointer_gen True --word_vocab_size 15000 --coverage True --convert_to_coverage_model False >logs/pgen10e_10cov

 ```
### Restore Pretrained Model and Run RL Training
You need to give the `exp_name` for restoring the pretained model, e.g. model_seq2seq/model_pgen10e_10cov
```
python3 run.py --mode rl_train  --reward_type bleu_4 --data_path ./data/ROC_data --exp_name=model_seq2seq >logs/seq2seq_rl
```
```
python3 run.py --mode=rl_train  --reward_type bleu_4 --data_path ./data/ROC_data --exp_name model_pgen10e_10cov >logs/pgen_rl
```

### Run Beam Search Decoding

```
python3 run.py --mode beam_search_decode --data_path ./data/ROC_data --exp_name model_seq2seq --decode_ckpt_dir eval_seq2seq
```
### Run Automatic Evaluation Script
```
cd evaluation
python2 evaluation_py2.py --data_path ./data/ROC_data --exp_name model_seq2seq --result_dir eval_seq2seq12600
```
