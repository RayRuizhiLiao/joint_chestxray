'''
Authors: Geeticka Chauhan, Ruizhi Liao

Parser for the arguments used by the joint model, including the data directory
'''

import argparse
import copy
import time 

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--text_data_dir', # this will need to be changed in the code
        default='/data/vision/polina/projects/chestxray/geeticka/bert/converted_data/',
        help='the input data directory; should contain the .tsv files (or other data files) for the task')
parser.add_argument('--img_data_dir', 
        #default = '/data/vision/polina/projects/chestxray/data_v2/npy/',
        default = '/data/vision/polina/scratch/ruizhi/chestxray/data/png_16bit',
        help='the input data directory; should contain the .tsv files (or other data files) for the task')
parser.add_argument('--img_localdisk_data_dir', 
        default = '/var/tmp/geeticka',
        help='the output image directory in the local drive')
parser.add_argument('--bert_pretrained_dir',
        default='/data/vision/polina/projects/chestxray/geeticka/bert/scibert/scibert_scivocab_uncased/',
        help='this is where BERT will look for pre-trained models to load parameters from')
parser.add_argument('--output_dir', # the fine tuned model directory, output of the training data
        default='/data/vision/polina/scratch/geeticka/chestxray_joint/experiments/',
        help='the output directory where the fine-tuned model and checkpoints will be written after training')
parser.add_argument('--reports_dir', # later change this just to use the output directory
        default='/data/vision/polina/scratch/geeticka/chestxray_joint/outputs/multilabel/',
        help='the directory where the evaluation reports will be written to')
parser.add_argument('--data_split_path', # information about the data split
        default='/data/vision/polina/projects/chestxray/work_space_v2/report_processing/edema_labels-12-03-2019/',
        help='the output directory where the different data splits will be written')
parser.add_argument('--joint_semisupervised_pretrained_checkpoint',
        default='/data/vision/polina/scratch/geeticka/chestxray_joint/experiments/cross_val/model/' \
                'semisupervised_phase1/ranking_training_12346_validation_5/checkpoints/checkpoint-250/',
        help='the checkpoint directory that the joint model will load from before training')


# training related parameters
parser.add_argument('--max_seq_length',
        default=320, type=int, help='maximum sequence length for bert')
parser.add_argument('--train_batch_size',
        default=4, type=int, help='train batch size')
parser.add_argument('--eval_batch_size',
        default=8, type=int, help='eval batch size')
parser.add_argument('--learning_rate',
        default=2e-5, type=float, help='Intial learning rate')
parser.add_argument('--num_train_epochs',
        default=300, type=int, help='number of epochs to train for')
parser.add_argument('--random_seed', # make this unchangeable
        default=42, type=int, help='random seed')
parser.add_argument('--gradient_accumulation_steps',
        default=1, type=int, help='gradient accumulation steps')
parser.add_argument('--warmup_proportion',
        default=0.1, type=float, help='warmup proportion')
parser.add_argument('--weight_decay',
        default=0.1, type=float, help='weight decay')
parser.add_argument('--config_name',
        default='bert_config.json', help='bert config file')
parser.add_argument('--weights_name',
        default='pytorch_model.bin', help='name of the weights file')
parser.add_argument('--max_grad_norm', default=1.0, type=float, 
        help='for gradient clipping')
parser.add_argument('--joint_loss_method', default='ranking',
        help='whether to use l2 similarity function, cosine similarity, dot product similarity or ranking-based loss')
parser.add_argument('--joint_loss_similarity_function', default='dot',
        help='which similarity function to be used for the ranking-based loss.'\
             'Options are l2, cosine, dot_product')
parser.add_argument('--scheduler', default='WarmupLinearSchedule', type=str, 
        help='The scheduler for learning rate during training')

#logging info, not hyperparams
parser.add_argument('--output_channel_encoding', default='multiclass', 
        help='whether to use multi-label (3 channel) vs multi-class (one hot) based classification')
parser.add_argument('--id', default='example',
        help='id to use for the outputs directory')
parser.add_argument('--data_split_mode', default='testing',
        help='whether to run in cross val or testing mode')
parser.add_argument('--use_text_data_dir', default=False, action='store_true',
        help='whether to use the given text data dir path; otherwise concatenate it with other tags')
parser.add_argument('--use_data_split_path', default=False, action='store_true',
        help='whether to use the given data split path; otherwise concatenate it with other tags')
parser.add_argument('--compute_auc', default=True,
        help='whether to compute auc for evaluation')
parser.add_argument('--compute_mse', default=True,
        help='whether to compute mse for evaluation')
parser.add_argument('--compute_accuracy_f1', default=True,
        help='whether to compute accuracy and f1 for evaluation')
parser.add_argument('--training_mode', default='supervised',
        help='whether to perform the supervised or semisupervised training,' \
             'you can specify one of the three options: supervised, semisupervised_phase1, semisupervised_phase2.' \
             'If semisupervised_phase2, the joint model will load from pretrained_model_checkpoint before training.')
parser.add_argument('--semisupervised_training_data', default='allCXR',
        help='whether to use allCXR or allCHF for semisupervised training.')
parser.add_argument('--bert_pool_last_hidden', default=False, action='store_true',
        help='whether to pool the full sequence of the last layer hidden states in bert model')
parser.add_argument('--bert_pool_use_img', default=False, action='store_true',
        help='whether to pool the image embedding to compute the attention score')
parser.add_argument('--bert_pool_img_lowerlevel', default=False, action='store_true',
        help='whether to pool the image embedding from the lower convolution layer')

parser.add_argument('--do_train', default=False, action='store_true', # need to do a test to make sure not false with training
        help='whether to perform training')
parser.add_argument('--do_eval', default=False, action='store_true', # need to do a test to make sure not false with eval
        help='whether to perform evaluation')
parser.add_argument('--use_masked_txt', default=False, action='store_true',
        help='whether to use masked reports')
parser.add_argument('--use_all_data', default=False, action='store_true',
        help='whether to use all the data for training or evaluation')
parser.add_argument('--use_pretrained_checkpoint', default=False, action='store_true',
        help='whether to use a pretrained checkpoint for initializing model parameters')
parser.add_argument('--print_predictions', default=False, action='store_true',
        help='whether to print predictions of each evaluation data point for image model evaluation')
parser.add_argument('--print_embeddings', default=False, action='store_true',
        help='whether to print embeddings of each evaluation data point')
parser.add_argument('--num_cpu_workers', default=8, 
        help='number of cpu cores being used')
#parser.add_argument('--development_or_test', default='development', required=True,
#        help='whether to use development or test data')
parser.add_argument('--logging_steps', default=50, type=int, 
        help='the number of steps for logging')
parser.add_argument('--save_epochs', default=1, type=int, 
        help='at which epochs to save the model checkpoints')
#parser.add_argument('--evaluate_during_training', default=False, action='store_true', 
#        help='whether to evaluate the model during training')
parser.add_argument('--eval_all_checkpoints', default=False, action='store_true', 
        help='whether to evaluate all model checkpoints')
parser.add_argument('--overwrite_output_dir', default=False, action='store_true', 
        help='whether to overwrite the output directory while training, depends on id and output_dir args')
parser.add_argument('--reprocess_input_data', default=False, action='store_true',
        help='whether to reprocess input data, stored as features in the data directory')
parser.add_argument('--training_folds', default=[1,2,3,4],
                    nargs='+', type=int, help="folds for training")
parser.add_argument('--validation_folds', default=[5],
                    nargs='+', type=int, help="folds for validation")

def get_args(): return parser.parse_args()
