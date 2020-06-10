'''
Author: Geeticka Chauhan, Ruizhi Liao
This file contains the scripts to do training and evaluation and other utilities used by the main
'''
import os
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.tensorboard import SummaryWriter #TODO: this will only work with Pytorch 1.0
from tqdm import tqdm, trange
from joint_img_txt.model import model_utils
from joint_img_txt.model import loss as custom_loss
from scripts import metrics as eval_metrics
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn import Softmax,LogSoftmax
from scipy.stats import logistic
from scipy.special import softmax

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import logging
import numpy as np
import json
import sklearn
import time


#def debug(args, device, model, tokenizer):
#    train_dataset, num_labels = model_utils.load_and_cache_examples(args, tokenizer, evaluate=False)
#    total_time = 0
#    for i in range(0, len(train_dataset)):
#        start = time.time()
#        item = train_dataset[i]
#        end = time.time()
#        print('end - start', end - start)
#        total_time += end - start
#    print('total time', total_time)

# The training function
def train(args, device, model, tokenizer):
    '''
    Create a logger and tensorboard writer
    '''
    logger = logging.getLogger(__name__)
    tb_writer = SummaryWriter(log_dir=args.tsbd_dir)

    '''
    Create a training dataset and dataloader
    '''
    train_dataset, num_labels = model_utils.load_and_cache_examples(args, 
                                                                    tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,
                                  num_workers=args.num_cpu_workers, 
                                  pin_memory=True)
    print("Length of the dataloader is ", len(train_dataloader))

    num_train_optimization_steps = len(train_dataloader) // \
        args.gradient_accumulation_steps * args.num_train_epochs
    print("Number of the total training steps = ", num_train_optimization_steps)

    '''
    Create an optimizer and a scheduler instance 
    '''
    # Below is a little complicated - 
    # they changed the implementation of the BertAdam to make it AdamW without
    # any gradient clipping so now you have to do your own. 
    # Read details at the bottom of readme at 
    # https://github.com/huggingface/pytorch-transformers#migrating-from-pytorch-pretrained-bert-to-pytorch-transformers
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
        'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0}
        ]
    args.warmup_steps = args.warmup_proportion * num_train_optimization_steps
    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=args.learning_rate,
                      correct_bias=False) 
    # eps = args.adam_epsilon (can define); correct_bias can be set 
    # to false like in the original tensorflow repository
    if args.scheduler == 'WarmupLinearSchedule':
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                         t_total=num_train_optimization_steps)
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-6)

    '''
    Log some important training parameters
    '''
    logger.info("***** Running training *****")
    logger.info("  Data split file: %s", args.data_split_path)
    logger.info("  Data split mode: %s", args.data_split_mode)
    logger.info("  Training fold = %s\t Validation fold = %s"%(args.training_folds, args.validation_folds))
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_train_optimization_steps)
    logger.info("  Initial learning rate = %f", args.learning_rate)
    logger.info("  Learning rate scheduler = %s", args.scheduler)
    logger.info("  Number of output channels = %s", num_labels)

    '''
    Train the model
    '''
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    last_epoch_loss = 0.0
    last_epoch_global_step = 0
    logging_img_loss, logging_txt_loss, logging_img_txt_loss, logging_joint_loss = 0.0, 0.0, 0.0, 0.0
    last_epoch_img_loss, last_epoch_txt_loss, last_epoch_img_txt_loss, last_epoch_joint_loss = 0.0, 0.0, 0.0, 0.0
    tr_img_loss, tr_txt_loss, tr_img_txt_loss, tr_joint_loss = 0.0, 0.0, 0.0, 0.0
    # https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/7
    # to check the purpose of zero-ing out the gradients between minibatches
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    # pay attention to why model.zero_grad above and model_train in the loop
    # loss is provided when labels is provided to the model
    # In my case I am not doing that
    model.train()
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        tr_epoch_loss = 0
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
            image, label_raw, txt_ids, txt_mask, txt_segment_ids, label_onehot_or_ordinal, report_id = batch
            # label_raw is always 0-3 and label_onehot_or_ordinal is one-hot or ordinal depending on whether
            # multiclass or multilabel
            # report_id is the radiology report study ID that's unique to each report

            inputs = {  'input_img':                image,
                        'input_ids':                txt_ids,
                        'attention_mask':           txt_mask,
                        'token_type_ids':           txt_segment_ids,
                        'labels':                   None,
                        'same_classifier':          args.share_img_txt_classifier,
                        'bert_pool_last_hidden':    args.bert_pool_last_hidden,
                        'bert_pool_use_img':        args.bert_pool_use_img,
                        'bert_pool_img_lowerlevel': args.bert_pool_img_lowerlevel} 

            # labels is None so that I can just get logits and apply loss here
            
            outputs = model(**inputs)
            img_embedding, img_logits, txt_embedding, txt_logits = outputs[:4]  
            # model outputs are always tuple in pytorch-transformers (see doc)
            
            if args.output_channel_encoding == 'multilabel' and args.training_mode != 'semisupervised_phase1':
                # Replace the image label with the ordinally encoded label
                label_ordinal = label_onehot_or_ordinal

                BCE_loss_criterion = BCEWithLogitsLoss()
                img_loss = BCE_loss_criterion(img_logits.view(-1, num_labels), 
                                              label_ordinal.view(-1, num_labels).float())
                txt_loss = BCE_loss_criterion(txt_logits.view(-1, num_labels), 
                                              label_ordinal.view(-1, num_labels).float())

            elif args.output_channel_encoding == 'multiclass' and args.training_mode != 'semisupervised_phase1':
                label = label_raw
                CrossEntropyCriterion = CrossEntropyLoss() # includes the softmax and only accepts label 0-3
                img_loss = CrossEntropyCriterion(img_logits.view(-1, num_labels),
                                                label.view(-1).long())
                txt_loss = CrossEntropyCriterion(txt_logits.view(-1, num_labels),
                                                label.view(-1).long())

            if args.use_imputed_labels:
                if args.output_channel_encoding == 'multilabel':
                    softmax = Softmax(dim=-1)
                    txt_probs = softmax(txt_logits)
                    BCE_loss_criterion = BCEWithLogitsLoss()
                    img_txt_loss = BCE_loss_criterion(img_logits.view(-1, num_labels),
                                                      txt_probs.view(-1, num_labels))
                elif args.output_channel_encoding == 'multiclass':
                    logSoftmax = LogSoftmax(dim=-1)
                    log_img_probs = logSoftmax(img_logits)
                    softmax = Softmax(dim=-1)
                    txt_probs = softmax(txt_logits)
                    img_txt_loss = custom_loss.dot_product_loss(log_img_probs,
                                                                txt_probs)

            if args.joint_loss_method == 'l2':
                joint_loss_criterion = torch.nn.MSELoss()
                joint_loss = joint_loss_criterion(img_embedding, txt_embedding)
            elif args.joint_loss_method == 'cosine':
                joint_loss_criterion = torch.nn.CosineEmbeddingLoss()
                y = torch.ones(img_embedding.shape[0], device=device) 
                # for each row in the batch make sure they are similar
                y.requires_grad = False
                joint_loss = joint_loss_criterion(img_embedding, txt_embedding, y) 
                # cosine loss needs to specify
                # whether looking for similarity between the tensors or not
            elif args.joint_loss_method == 'ranking':
                joint_loss = custom_loss.ranking_loss(img_embedding,
                                                      txt_embedding,
                                                      label_raw,
                                                      report_id,
                                                      similarity_function=args.joint_loss_similarity_function)
            elif args.joint_loss_method == 'dot':
                joint_loss = custom_loss.dot_product_loss(img_embedding,
                                                          txt_embedding)

            if args.training_mode == 'supervised' or args.training_mode == 'semisupervised_phase2':
                if args.use_imputed_labels:
                    loss = img_loss+txt_loss+joint_loss+img_txt_loss
                else:
                    loss = img_loss+txt_loss+joint_loss
                    img_txt_loss = joint_loss
            if args.training_mode == 'semisupervised_phase1':
                if args.use_imputed_labels:
                    loss = joint_loss+img_txt_loss
                else:
                    loss = joint_loss
                    img_txt_loss = joint_loss
                img_loss = joint_loss
                txt_loss = joint_loss

            print("\r%f" % loss, end='')
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if 'grad' in optimizer_grouped_parameters:
                torch.nn.utils.clip_grad_norm_(optimizer_grouped_parameters, 
                                               args.max_grad_norm)

            tr_loss += loss.item()
            tr_epoch_loss += loss.item()
            tr_img_loss += img_loss.item()
            tr_txt_loss += txt_loss.item()
            tr_img_txt_loss += img_txt_loss.item()
            tr_joint_loss += joint_loss.item() # TODO: convert loss to a tensor
            if epoch == args.num_train_epochs - 1:
                last_epoch_loss += loss.item()
                last_epoch_img_loss += img_loss.item()
                last_epoch_txt_loss += txt_loss.item()
                last_epoch_img_txt_loss += img_txt_loss.item()
                last_epoch_joint_loss += joint_loss.item() # TODO: convert loss to a tensor
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                # important: Pytorch 0.1 and above needs optimizer step to happen before
                # see https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
                # and notice this open bug with LR scheduler https://github.com/pytorch/pytorch/issues/22107
                if args.scheduler == 'WarmupLinearSchedule':
                    scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad() # optimizer has more parameters than my model, so I will follow this
                global_step += 1
                if epoch == args.num_train_epochs -1:
                    last_epoch_global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('learning_rate', 
                                         optimizer.param_groups[0]['lr'], 
                                         global_step)
                    tb_writer.add_scalar('loss/train', 
                                         (tr_loss - logging_loss)/args.logging_steps, 
                                         global_step)
                    tb_writer.add_scalar('loss_img/train', 
                                         (tr_img_loss - logging_img_loss)/args.logging_steps, 
                                         global_step)
                    tb_writer.add_scalar('loss_txt/train', 
                                         (tr_txt_loss - logging_txt_loss)/args.logging_steps, 
                                         global_step)
                    tb_writer.add_scalar('loss_joint/train', 
                                         (tr_joint_loss - logging_joint_loss)/args.logging_steps, 
                                         global_step)
                    image_grid = torchvision.utils.make_grid(image)
                    tb_writer.add_image('input_images', image_grid, global_step)
                    logger.info("  [%d, %5d, %5d] learning rate = %.7f"%\
                        (epoch + 1, step + 1, global_step,
                         optimizer.param_groups[0]['lr']))
                    logger.info("  [%d, %5d, %5d] loss = %.5f"%\
                        (epoch + 1, step + 1, global_step, 
                         (tr_loss - logging_loss)/args.logging_steps))
                    logger.info("  [%d, %5d, %5d] joint loss = %.5f"%\
                        (epoch + 1, step + 1, global_step, 
                         (tr_joint_loss - logging_joint_loss)/args.logging_steps))
                    logger.info("  [%d, %5d, %5d] image loss = %.5f"%\
                        (epoch + 1, step + 1, global_step, 
                         (tr_img_loss - logging_img_loss)/args.logging_steps))
                    logger.info("  [%d, %5d, %5d] text loss = %.5f"%\
                        (epoch + 1, step + 1, global_step, 
                         (tr_txt_loss - logging_txt_loss)/args.logging_steps))
                    logger.info("  [%d, %5d, %5d] image_text loss = %.5f"%\
                        (epoch + 1, step + 1, global_step, 
                         (tr_img_txt_loss - logging_img_txt_loss)/args.logging_steps))
                    logging_loss = tr_loss
                    logging_img_loss = tr_img_loss
                    logging_txt_loss = tr_txt_loss
                    logging_img_txt_loss = tr_img_txt_loss
                    logging_joint_loss = tr_joint_loss
        if args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(tr_epoch_loss)

        if args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0:
            # Save model checkpoint
            output_dir = os.path.join(args.checkpoints_dir, 'checkpoint-{}'.format(epoch+1))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)

        logger.info("  Epoch %d loss = %.5f" % (epoch + 1, tr_epoch_loss))

    return {'global_step': global_step,
            'training_loss': tr_loss / global_step,
            'training_img_loss': tr_img_loss / global_step, 
            'training_txt_loss': tr_txt_loss / global_step,
            'training_joint_loss': tr_joint_loss / global_step,
            'last_epoch_training_loss': last_epoch_loss / last_epoch_global_step,
            'last_epoch_img_loss': last_epoch_img_loss / last_epoch_global_step,
            'last_epoch_txt_loss': last_epoch_txt_loss / last_epoch_global_step,
            'last_epoch_img_txt_loss': last_epoch_img_txt_loss / last_epoch_global_step,
            'last_epoch_joint_loss': last_epoch_joint_loss / last_epoch_global_step}


# the evaluation script
def evaluate(args, device, model, tokenizer, eval_dataset, dump_prediction_files=False, prefix=""):
    logger = logging.getLogger(__name__)
    eval_output_dir = args.reports_dir

    results_txt = {} # this will just store the precision, recall, f1 of a single run of evaluate (for txt model)
    results_img = {}

    if not os.path.exists(eval_output_dir):
        print('The output directory %s does not exist!'%eval_output_dir)
        exit(0)
        #    os.makedirs(eval_output_dir)


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
            num_workers=args.num_cpu_workers, pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix)) # prefix refers to epoch
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    preds_txt = None
    preds_img = None
    out_labels = None
    report_ids = None
    img_embeddings = None
    txt_embeddings = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        image, label_raw, txt_ids, txt_mask, txt_segment_ids, label_onehot_or_ordinal, report_id = batch
        for j in range(len(label_raw)):
            if args.output_channel_encoding == 'multiclass':
                converted_label = torch.tensor(model_utils.convert_to_onehot(label_raw[j][0]), dtype=torch.long)
            if args.output_channel_encoding == 'multilabel':
                converted_label = torch.tensor(model_utils.convert_to_ordinal(label_raw[j][0]), dtype=torch.long)
            label_onehot_or_ordinal[j] = converted_label
            # we need the above conversion because label_raw is the one read from the csv file 
        label_onehot_or_ordinal.to(device=device)

        #image, label_raw, input_ids, input_mask, segment_ids, label_onehot_or_ordinal = batch
        with torch.no_grad():

            inputs = {  'input_img':                image,
                        'input_ids':                txt_ids,
                        'attention_mask':           txt_mask,
                        'token_type_ids':           txt_segment_ids,
                        'labels':                   None,
                        'same_classifier':          args.share_img_txt_classifier,
                        'bert_pool_last_hidden':    args.bert_pool_last_hidden,
                        'bert_pool_use_img':        args.bert_pool_use_img,
                        'bert_pool_img_lowerlevel': args.bert_pool_img_lowerlevel} 
            #inputs = {  'input_img':        image,
            #            'input_ids':        input_ids,
            #            'attention_mask':   input_mask,
            #            'token_type_ids':   segment_ids,
            #            'labels':           None,
            #            'same_classifier':  args.share_img_txt_classifier} # we purposefully make it none so that logits returned not loss
            outputs = model(**inputs)
            img_embedding, img_logits, txt_embedding, txt_logits = outputs[:4]

            #img_label = txt_labels_ordinal #Replace the image label with the ordinally encoded label
            #txt_labels = txt_labels_ordinal
            out_label = label_onehot_or_ordinal
            out_label_raw = label_raw
        nb_eval_steps += 1
        
        if (preds_txt is None  and preds_img is not None) or (preds_img is None and preds_txt is not None):
            raise Exception('No image or text prediction - this is not possible because they should be None at the same time')
        
        img_embedding_batch = img_embedding.detach().cpu().numpy()
        txt_embedding_batch = txt_embedding.detach().cpu().numpy()
        pred_txt_batch = txt_logits.detach().cpu().numpy() # shape: 10,3
        pred_img_batch = img_logits.detach().cpu().numpy()
        out_label_batch = out_label.detach().cpu().numpy()
        out_label_raw_batch = out_label_raw.detach().cpu().numpy()
        report_id_batch = report_id.detach().cpu()
        
        if preds_txt is None and preds_img is None:
            preds_txt = pred_txt_batch
            preds_img = pred_img_batch
            out_labels = out_label_batch # gold label of image and text is the same at the moment
            out_labels_raw = out_label_raw_batch
            report_ids = report_id_batch
            img_embeddings = img_embedding_batch
            txt_embeddings = txt_embedding_batch
        else:
            preds_txt = np.append(preds_txt, pred_txt_batch, axis=0)
            preds_img = np.append(preds_img, pred_img_batch, axis=0)
            out_labels = np.append(out_labels, out_label_batch, axis=0)
            out_labels_raw = np.append(out_labels_raw, out_label_raw_batch, axis=0)
            report_ids = np.append(report_ids, report_id_batch, axis=0)
            img_embeddings = np.append(img_embeddings, img_embedding_batch, axis=0)
            txt_embeddings = np.append(txt_embeddings, txt_embedding_batch, axis=0)

    if args.print_predictions:
        output_preds_file = os.path.join(eval_output_dir, "eval_results_images_preds.txt")
        with open(output_preds_file, "w") as writer:
            for i in range(len(report_ids)):
                writer.write('%s, '%report_ids[i])
                writer.write('%s\n'%preds_img[i])

    if args.print_embeddings:
        out_labels_raw_path = os.path.join(eval_output_dir, "eval_results_labels")
        np.save(out_labels_raw_path, out_labels_raw)
        img_embeddings_path = os.path.join(eval_output_dir, "eval_results_image_embeddings")
        np.save(img_embeddings_path, img_embeddings)
        txt_embeddings_path = os.path.join(eval_output_dir, "eval_results_text_embeddings")
        np.save(txt_embeddings_path, txt_embeddings)
        # with open(output_preds_file, "w") as writer:
        #     for i in range(len(report_ids)):
        #         writer.write('%s\n'%img_embeddings[i])
        #         writer.write('%s\n'%txt_embeddings[i])

    if args.compute_accuracy_f1:
        fed_labels = out_labels if args.output_channel_encoding == 'multilabel' else out_labels_raw
        acc_f1_txt, _, _ = eval_metrics.compute_acc_f1_metrics(fed_labels, preds_txt,
                                                                args.output_channel_encoding) # based on thresholding
        acc_f1_img, _, _ = eval_metrics.compute_acc_f1_metrics(fed_labels, preds_img,
                                                                args.output_channel_encoding) 
        results_txt.update(acc_f1_txt)
        results_img.update(acc_f1_img)

    if args.compute_auc:
        if args.output_channel_encoding == 'multilabel':
            preds_img_squashed = logistic.cdf(preds_img)
            preds_txt_squashed = logistic.cdf(preds_txt)
        else:
            preds_img_squashed = softmax(preds_img, axis=1) #TODO geeticka: check if the dimensions are right
            preds_txt_squashed = softmax(preds_txt, axis=1)

        auc_images, pairwise_auc_images = eval_metrics.compute_auc(out_labels, preds_img_squashed,
                args.output_channel_encoding)
        auc_txt, pairwise_auc_txt = eval_metrics.compute_auc(out_labels, preds_txt_squashed,
                args.output_channel_encoding)
        if args.output_channel_encoding == 'multiclass': # let's compute the 3 channel auc value
            ord_auc_images = eval_metrics.compute_ordinal_auc_from_multiclass(out_labels_raw, preds_img_squashed)
            ord_auc_txt = eval_metrics.compute_ordinal_auc_from_multiclass(out_labels_raw, preds_txt_squashed)
            results_img['ordinal_auc'] = ord_auc_images
            results_txt['ordinal_auc'] = ord_auc_txt

        results_img['auc'] = auc_images
        results_txt['auc'] = auc_txt
        results_img['pairwise_auc'] = pairwise_auc_images
        results_txt['pairwise_auc'] = pairwise_auc_txt

    if args.compute_mse:
        results_txt['mse'] = eval_metrics.compute_mse(preds_txt, out_labels, args.output_channel_encoding)
        results_img['mse'] = eval_metrics.compute_mse(preds_img, out_labels, args.output_channel_encoding)
    #print("Result img all egs", result_img_all_egs)
    #result_img = np.mean(result_img_all_egs)
    #print('Result img', result_img)


    if dump_prediction_files == True:
        #TODO: only do below for the main checkpoint
        for vals, variables in zip(['txt', 'img'],[[results_txt, out_labels, preds_txt],
            [results_img, out_labels, preds_img]]):
            output_eval_file = os.path.join(eval_output_dir, "eval_results_{}.txt".format(vals))
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} {} *****".format(vals, prefix))
                print('**** Eval results {} {} *****'.format(vals, prefix))
                for key in sorted(variables[0].keys()): # result file
                    logger.info("  %s = %s"%(key, str(variables[0][key])))
                    print('   %s = %s'%(key, str(variables[0][key])))
                    writer.write("%s = %s\n" % (key, str(variables[0][key])))
            with open(os.path.join(eval_output_dir, 'gold_labels_{}.txt'.format(vals)), 'w') as gold_file:
                for item in variables[1]: # gold labels
                    gold_file.write('%s\n'%item)
            with open(os.path.join(eval_output_dir, 'predicted_labels_{}.txt'.format(vals)), 'w') as pred_file:
                for item in variables[2]:
                    pred_file.write('%s\n'%item)

    return results_txt, results_img

def to_json_string(dictionary):
        """Serializes this instance to a JSON string."""
        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

def to_json_file(dictionary, json_file_path):
    """ Save this instance to a json file."""
    with open(json_file_path, "w", encoding='utf-8') as writer:
        writer.write(to_json_string(dictionary))
