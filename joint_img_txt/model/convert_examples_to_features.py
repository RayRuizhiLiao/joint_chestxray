'''
Authors: Geeticka Chauhan, Ruizhi Liao

This script contains functions for processing raw text.
Adapted from 
https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
'''
from multiprocessing import Pool, cpu_count
from tqdm import tqdm as tqdm

class InputFeatures(object):
    """A single set of features of text data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, report_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.report_id = report_id

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """
    converts examples to features
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    label_map['-1'] = -1 # To handle '-1' label (i.e. unlabeled data)
    examples_for_processing = [(example, label_map, max_seq_length, tokenizer) \
        for example in examples]
    process_count = cpu_count() - 1
    with Pool(process_count) as p:
            features = list(tqdm(p.imap(convert_example_to_feature, 
                                 examples_for_processing), 
                            total=len(examples)))
    return features

def convert_example_to_feature(example_row):
    """ 
    returns example_row
    """
    example, label_map, max_seq_length, tokenizer = example_row

    #TODO: geeticka don't need to change the output mode
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    #if output_mode == "classification":
    label_id = label_map[example.labels]
    #elif output_mode == "regression":
    #    label_id = float(example.label)
    #else:
    #    raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_id=label_id, 
                         report_id=example.report_id)


class InputFeaturesMultiLabel(object):
    """A single set of features of text data 
    in the case of multilabel.
    """

    def __init__(self, input_ids, input_mask, segment_ids, label_id, report_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.report_id = report_id

# for the multilabel case, convert examples to features
def convert_examples_to_features_multilabel(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    label_map['-'] = -1 # To handle '-1' label (i.e. unlabeled data)
    examples_for_processing = [(example, label_map, max_seq_length, tokenizer) \
        for example in examples]
    process_count = cpu_count() - 1
    with Pool(process_count) as p:
            features = list(tqdm(p.imap(convert_example_to_feature_multilabel, 
                                        examples_for_processing), 
                            total=len(examples)))
    return features

def convert_example_to_feature_multilabel(example_row):
    # return example_row
    example, label_map, max_seq_length, tokenizer = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding # this refers to ids for the embedding dictionary
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = [label_map[label] for label in example.labels]
    # To handle '-1' label (i.e. unlabeled data)
    if -1 in label_id:
        label_id = [-1, -1, -1]

    return InputFeaturesMultiLabel(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id, 
        report_id=example.report_id)
