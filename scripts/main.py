'''
Authors: Geeticka Chauhan, Ruizhi Liao
Main script to run training and evaluation of the image-text joint model
for estimating the severity of pulmonary edema on MIMIC-CXR
'''
import glob
import pickle
import os
import sys
from pathlib import Path
from tqdm import tqdm_notebook, trange
from multiprocessing import Pool, cpu_count
import logging
import time
import uuid # For generating a unique id

current_path = os.path.dirname(os.path.abspath(__file__))
current_path = Path(current_path)
parent_path = current_path.parent
print('Project home directory: ', str(parent_path))
sys.path.insert(0, str(parent_path)) # Do not use sys.path.append here
print('sys.path: ', sys.path)

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from joint_img_txt.model.model_utils import *
from joint_img_txt.model import model_utils
from joint_img_txt.model import convert_examples_to_features
from joint_img_txt.model.model import ImageTextModel
from scripts import main_utils, parser


def main():
    args = parser.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert torch.cuda.is_available(), "No GPU/CUDA is detected!"
    # Training on CPU is almost infeasible,
    # but evaluation/inference can be done on CPU

    '''
    Do initial argument checks
    '''
    if args.id == 'dummy': 
        args.id = str(uuid.uuid4())
        # If no ID is specified,
        # then we will generate a radom ID as the folder name of this run

    if args.training_mode != 'supervised' and \
            args.training_mode != 'semisupervised_phase1' and \
            args.training_mode != 'semisupervised_phase2':
        raise Exception('You can do either supervised or semisupervised training!')
        # 'semisupervised_phase1' is essentially unsupervised learning of the joint model
        # on chest radiographs and radiology reports
        # 'semisupervised_phase2' is supervised learning with the initialization 
        # from the training results of semisupervised_phase1. 

    if args.semisupervised_training_data != 'allCXR' and \
            args.semisupervised_training_data != 'allCHF':
        raise Exception('You can train the model on all MIMIC-CXR (allCXR) or \
            the congestive heart failure cohort (allCHF)')

    if args.training_mode == 'semisupervised_phase2':
        if not os.path.isdir(args.joint_semisupervised_pretrained_checkpoint):
            raise Exception('The joint_semisupervised_pretrained_checkpoint directory has to exist in semisupervised_phase2')

    if args.output_channel_encoding != 'multilabel' and args.output_channel_encoding != 'multiclass':
        raise Exception('You can only select multilabel or multiclass classification')
    
    if args.data_split_mode != 'cross_val' and args.data_split_mode != 'testing':
        raise Exception('You can only give the options cross_val or testing for the args.data_split_mode')
    '''
    Setting the data split file
    '''

    if args.training_mode == 'supervised' or args.training_mode == 'semisupervised_phase2':
        data_split_file_postfix = ''
    elif args.semisupervised_training_data == 'allCHF':
        data_split_file_postfix = '-allCHF'
    elif args.semisupervised_training_data == 'allCXR':
        data_split_file_postfix = '-allCXR'

    if args.data_split_mode == 'testing' and args.do_eval:
        args.data_split_path = os.path.join(args.data_split_path, 
            'mimic-cxr-sub-img-edema-split-manualtest.csv')
    else:
        args.data_split_path = os.path.join(args.data_split_path, 
            'mimic-cxr-sub-img-edema-split{}.csv'.format(data_split_file_postfix))

    if args.joint_loss_method != 'l2' and args.joint_loss_method != 'cosine' \
        and args.joint_loss_method != 'dot' and args.joint_loss_method != 'ranking':
        raise Exception('You can either have l2, cosine, dot or ranking' \
            'as the joint loss calculation between the img-txt embedding')

    if args.joint_loss_similarity_function != 'l2' and args.joint_loss_similarity_function != 'cosine' \
        and args.joint_loss_similarity_function != 'dot':
        raise Exception('You can either have l2, cosine, or dot ' \
            'as the similarity function for the ranking loss in the img-txt embedding. You had %s'%args.joint_loss_similarity_function)

    '''
    Setting the directory structure and testing for arguments
    '''
    if args.use_masked_txt:
         args.text_data_dir = os.path.join(args.text_data_dir, 'masked')
    args.text_data_dir = os.path.join(args.text_data_dir, args.output_channel_encoding)

    if args.share_img_txt_classifier:
        args.model = 'model_same_classifier'
    else:
        args.model = 'model'

    if args.copy_data_to_local and args.copy_zip_to_local:
        raise Exception('Cannot copy all files and zip to local at the same time')

    if args.cache_images and (args.copy_data_to_local or args.copy_zip_to_local):
        raise Exception('Cannot cache images and copy images to local at the same time')

    if args.training_mode == 'supervised' or args.training_mode == 'supervised_masking':
        args.text_data_dir = os.path.join(args.text_data_dir, 'supervised', 'full')
        args.output_dir = os.path.join(args.output_dir, args.data_split_mode,
                args.model, args.training_mode, args.id)
    elif 'semisupervised' in args.training_mode:
        args.text_data_dir = os.path.join(args.text_data_dir, 'semisupervised',
                args.semisupervised_training_data, 'full')
        args.output_dir = os.path.join(args.output_dir, args.data_split_mode, args.model, 
                args.training_mode, args.semisupervised_training_data, args.id) 

    args.reports_dir = os.path.join(args.output_dir, 'eval_reports')
    args.tsbd_dir = os.path.join(args.output_dir, 'tsbd_dir')
    args.checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
        and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty." \
            " Use".format(args.output_dir)+" --overwrite_output_dir to overcome.")

    if not args.do_train and not args.do_eval:
        raise Exception('Either do_train or do_eval flag must be set as true')

    '''
    After argument checks, create the necessary directories. 
    Make sure no argument updating after this point.
    '''
    directories = [args.output_dir, args.reports_dir, args.tsbd_dir, args.checkpoints_dir]
    for directory in directories:
        if not(os.path.exists(directory)):
            os.makedirs(directory)
    if not os.path.exists(args.data_split_path):
        raise Exception('The data split path %s does not exist! Please check'%args.data_split_path)
    
    if args.do_eval:
        args.reports_dir = os.path.join(args.reports_dir,
            'eval_report_{}'.format(len(os.listdir(args.reports_dir))))
        if not os.path.exists(args.reports_dir):
            os.makedirs(args.reports_dir)
        main_utils.to_json_file(vars(args), os.path.join(args.reports_dir, 'eval_args.json'))
        print('Location of the reports directory %s'%args.reports_dir)

    print('Classification type: {}'.format(args.output_channel_encoding)) # whether multilabel (3 channel) or multiclass
    print('Loss method in the image-text embedding space: {}'.format(args.joint_loss_method)) 
    if args.joint_loss_method == 'ranking':
        print('Similarity function for the ranking loss in the img-txt embedding:', args.joint_loss_similarity_function)
    print('Currently doing **{}**'.format(args.data_split_mode)) # prev args.development_or_test
    print('Training mode: {}'.format(args.training_mode))
    print('Training is {}'.format(args.do_train))
    print('Eval is {}'.format(args.do_eval))
    print('Caching the images to RAM: {}'.format(args.cache_images))
    print('Copying the images to local disk (/var/tmp/cxr_data/): {}'.format(args.copy_data_to_local))
    print('Copying the zip to local disk (/var/tmp/zip_cxr_data/): {}'.format(args.copy_zip_to_local))
    print('Using png images: {}'.format(args.use_png))
    print('Cuda is available: {}'.format(torch.cuda.is_available()))
    print('Device is being used: ', device)
    print('Scheduler used: ', args.scheduler)
    print('Learning Rate: ', args.learning_rate)
    print('Training epochs: ', args.num_train_epochs)
    print('Sharing the image and text classifier: ', args.share_img_txt_classifier)
    print('Text data directory: ', args.text_data_dir)
    if 'semisupervised' in args.training_mode:
        print('Semisupervised training directory: ', args.semisupervised_training_data)
    print('Using all Sequences in BERT last layer rather than just [CLS]: ', args.bert_pool_last_hidden)
    if args.bert_pool_last_hidden:
        print('Using img embedding for computing attention scores: ', args.bert_pool_use_img)
    '''
    Please note: for training, the pretrained model directory is args.bert_pretrained_dir
    and for evaluation, the pretrained model directory is args.output_dir
    '''
    if args.do_train:
        args.tsbd_dir = os.path.join(args.tsbd_dir, 
            'tsbd_{}'.format(len(os.listdir(args.tsbd_dir))))
        if not os.path.exists(args.tsbd_dir):
            os.makedirs(args.tsbd_dir)
        print('Location of the tensorboard directory %s'%args.tsbd_dir)
        log_file = os.path.join(args.output_dir, 'training.log')
        print('Pretrained model:\t {}'.format(args.bert_pretrained_dir))
    if args.do_eval:
        log_file = os.path.join(args.reports_dir, 'evaluating.log')
    print('Logging in:\t {}'.format(log_file))
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', 
        format='%(asctime)s - %(name)s %(message)s', datefmt='%m-%d %H:%M')
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrained_dir)
    # It is always loaded from bert_pretrained_dir
    # tokenizer is not something that constantly needs to be saved 
    # because only the pre-trained model determines this.

    '''
    Train or evaluate the model
    '''
    if args.do_train:
        start_time = time.time()
        logger = logging.getLogger('pytorch_transformers.modeling_utils').setLevel(logging.INFO)

        config = BertConfig.from_json_file(os.path.join(args.bert_pretrained_dir, args.config_name))
        config.num_labels = 3 if args.output_channel_encoding == 'multilabel' else 4
        if args.training_mode == 'semisupervised_phase2':
            model = ImageTextModel.from_pretrained(args.joint_semisupervised_pretrained_checkpoint)
            print('Pretrained model:\t {}'.format(args.joint_semisupervised_pretrained_checkpoint))
        elif args.use_pretrained_checkpoint:
            model = ImageTextModel.from_pretrained(args.joint_semisupervised_pretrained_checkpoint)
            print('Pretrained model:\t {}'.format(args.joint_semisupervised_pretrained_checkpoint))            
        else:
            model = ImageTextModel(config=config, pretrained_bert_dir=args.bert_pretrained_dir)
            print('Pretrained model:\t {}'.format(args.bert_pretrained_dir))

        model.to(device)
        loss_info = main_utils.train(args, device, model, tokenizer)
        # Reset the logger now
        logger = logging.getLogger(__name__)
        logger.info("Saving model checkpoint to %s", args.output_dir)
          
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model

        # new way of saving
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        main_utils.to_json_file(loss_info, os.path.join(args.output_dir, 'loss_info.json'))
        end_time = time.time()
    
    results_txt = {}
    results_img = {}
    losses_info = {}
    # eval should assume that the train ids already contain the necessary folders 
    # will deal with this later. Just copy eval images here 
    if args.do_eval:
        start_time = time.time()
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints: #TODO: have the ability to have another flag to control whether to 
            # print all evaluation numbers, and whether to write them all into reports - actually
            # should just have a logging for that
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' +
                args.weights_name, recursive=True)))
        logger = logging.getLogger(__name__)
        logger.info("Evaluate %d checkpoints ", len(checkpoints))
        eval_dataset, _ = model_utils.load_and_cache_examples(args, tokenizer)
        for checkpoint in checkpoints:
            #logger = logging.getLogger('pytorch_transformers.modeling_utils').setLevel(logging.WARN)
            epoch_number = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            logger = logging.getLogger('joint_img_txt.model.model').setLevel(logging.INFO)
            model = ImageTextModel.from_pretrained(checkpoint)
            model.to(device)
            dump_prediction_files = False
            if checkpoint == args.output_dir:
                dump_prediction_files = True
                epoch_number = 'final' # fix the epoch number for the final dumping
            print('***    Epoch {}'.format(epoch_number))
            print('\t\t\t Checkpoint: {}'.format(checkpoint))
            result_txt, result_img = main_utils.evaluate(args, device, model, tokenizer, eval_dataset,
                    dump_prediction_files, prefix=epoch_number)
            result_txt = dict((k + '_{}'.format(epoch_number), v) for k, v in result_txt.items())
            result_img = dict((k + '_{}'.format(epoch_number), v) for k, v in result_img.items())
            #loss_info = dict((k + '_{}'.format(epoch_number), v) for k, v in loss_info.items())
            results_txt.update(result_txt)
            results_img.update(result_img)
            #losses_info.update(loss_info)

        #logger = logging.getLogger(__name__)
        main_utils.to_json_file(results_txt, os.path.join(args.reports_dir, 'results_txt.json'))
        main_utils.to_json_file(results_img, os.path.join(args.reports_dir, 'results_img.json'))
        #main_utils.to_json_file(losses_info, os.path.join(args.reports_dir, 'losses_info.json'))
        end_time = time.time()
        

    print("\n\nTotal time to run:", round((end_time-start_time)/3600.0, 2))
    # in case this code is buggy, throw it at the very end
    if args.copy_data_to_local:
        local_images = LocalDiskData('', args.use_png, '', args.img_localdisk_data_dir,
                args.id) # this is fine i just want to delete folder
        local_images.delete_folder()
    if args.copy_zip_to_local:
        local_images = LocalDiskZipData('', args.use_png, args.img_localdisk_data_dir, 
                args.id)
        local_images.delete_zip_folder()


if __name__ == '__main__':
    main()
