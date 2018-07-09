import argparse
import numpy as np
from nlgeval import compute_metrics

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='script for evaluation')
    parser.add_argument('--data_path', default='../data/ROC_data',
                        help='Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
    parser.add_argument('--exp_name', default='0201',
                        help='Name for experiment. Logs will be saved in a directory with this name, under log_root.')
    parser.add_argument('--result_dir', default='eval_seq2seq15200', help='directory of test_pred to save.')
    args = parser.parse_args()
    argparams = vars(args)


    ckpt_name = 'BS_test_data-ckpt_from-' + argparams['result_dir']
    test_genarated_file = '../{}/{}/{}'.format(argparams['exp_name'], ckpt_name, 'test_data_results.txt')
    test_plot_file = '{}/{}'.format(argparams['data_path'], 'test_plot.txt')
    test_ending_file = '{}/{}'.format(argparams['data_path'], 'test_ending.txt')


    print('compute word overlap scores between target ending and generated ending')
    metrics_dict_target_generated = compute_metrics(hypothesis=test_genarated_file, references=[test_ending_file])
    print('-' * 50)
    # print('compute similarity scores between generated ending and target_ending')
    # metrics_dict_plot_generated = compute_metrics(hypothesis=test_genarated_file, references=[test_ending_file],
    #                                               no_overlap=True)
    # print('-' * 50)
    print('compute similarity scores between generated ending and plot')
    metrics_dict_plot_target = compute_metrics(hypothesis=test_genarated_file, references=[test_plot_file],
                                               no_overlap=True)
    # print('-' * 50)




