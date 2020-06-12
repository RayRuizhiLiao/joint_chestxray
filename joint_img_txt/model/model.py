'''
Authors: Geeticka Chauhan, Ruizhi Liao
This script contains the image-text joint model
that encodes image and text features in a joint embedding space.
Two classifiers with the same network architecture perform classification
on the image feature and the text feature respectively.
'''
import os
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import BertPreTrainedModel, PretrainedConfig
from pytorch_transformers import BertModel


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ImageResNet(nn.Module):

    def __init__(self, block, layers, output_channels=3, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ImageResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
        # if replace_stride_with_dilation is None:
        #     # each element in the tuple indicates if we should replace
        #     # the 2x2 stride with a dilated convolution instead
        #     replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
        #     raise ValueError("replace_stride_with_dilation should be None "
        #                      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 8, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 128, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 192, layers[5], stride=2)
        self.layer7 = self._make_layer(block, 192, layers[6], stride=2)
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc1 = nn.Linear(768, output_channels)
        #self.fc1 = nn.Linear(768, 24)
        #self.bn2 = nn.BatchNorm1d(24)
        #self.fc2 = nn.Linear(24, output_channels)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, same_classifier=False, use_lowerlevel_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # 8, 8, 512, 512
        #x = self.maxpool(x)

        x = self.layer1(x) # 8, 8, 256, 256
        x = self.layer2(x) # 8, 16, 128, 128
        x = self.layer3(x) # 8, 32, 64, 64
        x = self.layer4(x) # 8, 64, 32, 32
        x = self.layer5(x) # 8, 128, 16, 16
        x = self.layer6(x) # 8, 192, 8, 8
        lowerlevel_img_feat = x 
        x = self.layer7(x) # 8, 192, 4, 4
        
        x = self.avgpool(x)
        z = torch.flatten(x, 1) # 8, 768
        #x = self.fc1(z)
        #x = self.bn2(x)
        #x = self.relu(x)
        #y = self.fc2(x)
        outputs = (z,)
        if not same_classifier:
            logits = self.fc1(z)
            outputs += (logits,)
        if use_lowerlevel_features:
            outputs += (lowerlevel_img_feat,)
        return outputs # z, (logits), (layer6 output)


# Adapted from
# https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d
class TextBertForSequenceClassification(BertPreTrainedModel):
    """
    BERT model for multilabel/multiclass classification. Here we will treat the problem with 
    ordinal encoding. Note: because we are doing ordinal encoding, num_labels is actually 3 
    and not 4. So make sure to feed num_labels - 1 to this one. 
    """
    def __init__(self, config):
        super(TextBertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.pooler = TextBertAttentionPooler(config)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, same_classifier=False, use_all_sequence=False,
                img_embedding=None, output_img_txt_attn=False):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # When invesigating what is going on, use_all_sequence needs to be investigated further
        # because the hidden states of all the tokens seemed to be the same
        # print('CLS output', outputs[1])
        # print('CLS input', outputs[0][:,0])
        # print('Hidden states', outputs[0])
        if not use_all_sequence:
            pooled_output = outputs[1] # this is the default pooled output i.e. [CLS]
        else:
            # insert own pooling mechanism
            hidden_states = outputs[0] # now we have the whole BERT sequence to use i.e. max_seq_len
            # don't use the [CLS] and [SEP]
            # need to implement something called attention pooling over here
            pooled_output = self.pooler(input_ids, hidden_states, attention_mask, img_embedding,
                    output_img_txt_attn)
            if output_img_txt_attn:
                img_txt_attn = pooled_output[1]
                pooled_output = pooled_output[0]
            else:
                pooled_output = pooled_output[0]

        pooled_output = self.dropout(pooled_output)
        
        if same_classifier:
            outputs = (pooled_output,) + outputs[2:]
        else:
            logits = self.classifier(pooled_output)
            outputs = (pooled_output, logits,) + outputs[2:]
            # add hidden states and attention if they are here
        if use_all_sequence and output_img_txt_attn:
            outputs = outputs + (img_txt_attn,)
        return outputs  # pooled_output, (logits), (hidden_states), (txt_attentions), (img_txt_attn)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encode(self):
        for param in self.bert.parameters():
            param.requires_grad = True


# Bert Attention pooler based on simple pooler in hugging face repo
# ref:
# https://github.com/huggingface/transformers/blob/b33a385091de604afb566155ec03329b84c96926/pytorch_transformers/modeling_bert.py#L455
class TextBertAttentionPooler(nn.Module):
    def __init__(self, config):
        super(TextBertAttentionPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.attention_head_size = int(config.hidden_size) # we only have 1 attention head

    def forward(self, input_ids, hidden_states, attention_mask, img_embedding=None, output_img_txt_attn=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_bert.py L 688
        # since attention_mask is 1.0 for positions we want to attend and 0.0 for masked positions, this 
        # operation will create a tensor which is 0.0 for positions we want to attend and -10000.0 for masked
        # positions. Since we are adding it to the raw scores before the softmax, this is effectively the same
        # as removing these entirely. 
        compatible_attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        compatible_attention_mask = (1.0 - compatible_attention_mask) * -10000.0

        # need to compute the attention scores somehow. In the simplest case, randomly initialize them and 
        # connect them by doing that weighted sum with the hidden states to compute the pooled representation
        # for the radiology report. Need to look at good ways to do attention weighted pooling when we 
        # have multiple sentences - something hierarchical would be interesting. Interesting after idea
        # how about including position like encoding to highlight when we are in a new sentence vs not 
        # can be as simple as 0 or 1 segment embeddings the way bert does :D :D 
        #attention_scores = torch.empty(attention_mask.size,
        #        requires_grad=True).type(next(self.parameters()).dtype)
        #nn.init.ones_(attention_scores) # give equal attention score to all the tokens
        if img_embedding is None:
            attention_scores = torch.ones_like(compatible_attention_mask, requires_grad=True)
            hidden_states_used = hidden_states
        elif len(img_embedding.size()) == 2:
            # compute the dot product similarity
            # attention_scores = torch.matmul(img_embedding, hidden_states.transpose(-1, -2))
            # Above wouldn't work, gives the output as a 3D tensor but we need a 2D tensor with unsqueezed 
            img_embed_expanded = img_embedding.unsqueeze(-2) # from [batch,768] -> [batch, 1, 768]
            # compute the dot product
            print(hidden_states) # the hidden states look exactly the same
            print('Size of hidden states', hidden_states.size())
            attention_scores_expanded = torch.matmul(img_embed_expanded, hidden_states.transpose(-1,-2))
            attention_scores = attention_scores_expanded.squeeze()
            # from [batch, 1, token_sequences] -> [batch, token_sequences]
            # might need to look at
            # https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_bert.py
            # L 308 to look at the detail for the attention score reweighting
            hidden_states_used = hidden_states
        else: 
            # in this case we have img_embedding size as a 3D vector (not including batch), need to manipulate
            '''
            if the img_embedding is 3 dimensional then we compute the similarity based on image regions
            '''
            # let us first convert an img_embed from batch, channels, H, W to batch, images, channels
            # basically from 8, 192, 8, 8 to 8, 16, 768
            for x in range(0, img_embedding.size()[2], 2):
                for y in range(0, img_embedding.size()[2], 2):
                    flattened = torch.flatten(img_embedding[: , : , x:x+2 , y:y+2], 1)
                    flattened = flattened.unsqueeze(1)
                    if x == 0 and y == 0:
                        img_embed_expanded = flattened
                    else:
                        img_embed_expanded = torch.cat((img_embed_expanded, flattened), 1)
            print(img_embedding)
            print(img_embed_expanded)
            attention_scores = torch.matmul(img_embed_expanded, hidden_states.transpose(-1,-2))
            compatible_attention_mask = \
                    compatible_attention_mask.unsqueeze(-2).expand(attention_scores.size())
            hidden_states_used = hidden_states.unsqueeze(-3).expand(attention_scores.size()+\
                    (hidden_states.size()[-1],)) 
            # expand from batch, token_seq_len, 768 ---> batch, img_regions, token_seq_len, 768


        print('Before scaling', attention_scores)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        print('After scaling', attention_scores)
        attention_scores = attention_scores + compatible_attention_mask # batch_size X max_seq_len
        print('After adding mask', attention_scores)
        print('Attention scores size', attention_scores.size())
        
        # Normalize the attention scores to probabilities 
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # take softmax along token sequences dim
        print('Attention probs size', attention_probs.size())
        print('Attention_probs', attention_probs)
        # idk why it just became the same probability; that's a major bug
        # The transformer paper also drops out attention probabilities but we are not going to do that

        # Take the dot product between attention probs and hidden states to get the pooled output 
        '''
        if attention_probs = [[1,2,3],[4,5,6]] shape is 2,3
        hidden = [[[1,2],[1,2],[1,2]],[[1,2],[1,2],[1,2]]] shape is 2,3,2
        then we expand attention_probs to be [[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]] so that we 
        can do element wise product with hidden (shape 2,3,2) and then sum along the middle axis 
        to get something of the shape 2x2. Middle axis represents our token dimensions which we 
        want to collapse
        '''

        attention_probs_expanded = attention_probs.unsqueeze(-1).expand(hidden_states_used.size())
        pooled_output = torch.sum(attention_probs_expanded * hidden_states_used, -2) # collapse token_seq_len
        if img_embedding is not None and len(img_embedding.size()) > 2:
            #pooled_output = torch.mean(pooled_output, -2) # could do a max over this dimension alternatively
            pooled_output = torch.max(pooled_output, -2).values # torch.max returns values and indices
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        if output_img_txt_attn:
            return (pooled_output, attention_probs)
        return (pooled_output,)


class ImageTextModel(nn.Module):
    """
    BERT model for multi label classification. Here we will treat the problem with 
    ordinal encoding. Note: because we are doing ordinal encoding, num_labels is actually 3 
    and not 4. So make sure to feed num_labels - 1 to this one. 
    """
    def __init__(self, config, pretrained_bert_dir=None, block=BasicBlock, 
                 layers=[2, 2, 2, 2, 2, 2, 2], zero_init_residual=False, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):

        super(ImageTextModel, self).__init__()
        if pretrained_bert_dir != None:
            self.text_model =\
            TextBertForSequenceClassification.from_pretrained(pretrained_bert_dir,
                                                              config=config)
        else:
            self.text_model = TextBertForSequenceClassification(config=config)

        self.img_model = ImageResNet(block=block, 
                                     layers=layers, 
                                     output_channels=config.num_labels, 
                                     # image and text model should have same number of channels
                                     zero_init_residual=zero_init_residual, 
                                     groups=groups, 
                                     width_per_group=width_per_group, 
                                     replace_stride_with_dilation=replace_stride_with_dilation,
                                     norm_layer=norm_layer)


        self.config = config 
        # Bert will throw an error if config is not in the right format
        
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_img, input_ids, token_type_ids=None, attention_mask=None, 
                labels=None, position_ids=None, head_mask=None, same_classifier=False,
                bert_pool_last_hidden=False, bert_pool_use_img=False, 
                bert_pool_img_lowerlevel=False, output_img_txt_attn=False):
        '''
        We will get the output of the image model first to get the image embedding
        '''
        outputs_img = self.img_model.forward(input_img, same_classifier, 
                bert_pool_last_hidden and bert_pool_use_img and bert_pool_img_lowerlevel)
        z_img = outputs_img[0] # 8, 768
        if same_classifier:
            logits_img = self.classifier(z_img)
        else:
            logits_img = outputs_img[1]
        if bert_pool_last_hidden and bert_pool_use_img and bert_pool_img_lowerlevel:
            lowerlevel_img_feat = outputs_img[2] # 8, 192, 8, 8

        ''' 
        Next let's feed the image model output to the text model
        '''
        img_embedding = None
        if bert_pool_use_img and bert_pool_img_lowerlevel:
            img_embedding = lowerlevel_img_feat
        elif bert_pool_use_img and not bert_pool_img_lowerlevel:
            img_embedding = z_img
        outputs_txt = self.text_model.forward(input_ids=input_ids,
                                              token_type_ids=token_type_ids,
                                              attention_mask=attention_mask,
                                              labels=labels,
                                              position_ids=position_ids,
                                              head_mask=head_mask, 
                                              same_classifier=same_classifier,
                                              use_all_sequence=bert_pool_last_hidden,
                                              img_embedding= img_embedding, 
                                              output_img_txt_attn=output_img_txt_attn)

        z_txt = outputs_txt[0]
        if same_classifier:
            #fill in
            logits_txt = self.classifier(z_txt)
        else:
            logits_txt = outputs_txt[1]

        outputs = (z_img, logits_img, z_txt, logits_txt)
        if self.config.output_attentions and output_img_txt_attn:
            outputs = outputs + (outputs_txt[-2], outputs_txt[-1]) # txt_attn, img_txt_attn 
        elif self.config.output_attentions or output_img_txt_attn:
            outputs = outputs + (outputs_txt[-1],)

        return outputs # z_img, logits_img, z_txt, logits_txt, (txt_attn), (img_txt_attn)
    
    # based on https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    def save_pretrained(self, save_directory): # taken from huggingface, pretrained transformers
        """ 
        Save a model with its configuration file to a directory, so that it
        can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, 'pytorch_model.bin')

        torch.save(model_to_save.state_dict(), output_model_file)
    
    # based on https://github.com/huggingface/transformers/blob/v1.0.0/pytorch_transformers/modeling_utils.py
    @classmethod
    def from_pretrained(cls, pretrained_model_path, *inputs, **kwargs):
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Load config
        if config is None:
            config = PretrainedConfig.from_pretrained(pretrained_model_path, *inputs, **kwargs)
        if os.path.isdir(pretrained_model_path):
            archive_file = os.path.join(pretrained_model_path, 'pytorch_model.bin')
        else:
            raise Exception('Please provide a directory to load the model from, currently given',
                    pretrained_model_path)

        logger = logging.getLogger(__name__)
        print("Loading the image-text model")
        logger.info('Loading the image-text model')
        # Instantiate the model
        model = cls(config=config)

        # if the user has not provided the ability to load in their own state dict, but our module
        # in this case it is easier to just use save_pretrained and from_pretrained to read that 
        # saved checkpoint fully
        if state_dict is None:
            state_dict = torch.load(archive_file, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        
        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if hasattr(model, 'tie_weights'):
                    model.text_model.tie_weights()  # make sure word embedding weights are still tied
        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model
