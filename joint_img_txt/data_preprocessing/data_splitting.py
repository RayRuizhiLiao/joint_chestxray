'''
This script will read in the csv train and test files and generate the train and test and 
train and dev csv files
This generates files that are inputs for the bert model
Author: Geeticka Chauhan
'''

import os
import sys
import chestxray_joint.data.text.utils as utils
import random
in_dir = '/data/vision/polina/projects/chestxray/geeticka/pre-processed/reports_normalized'
out_dir = '/data/vision/polina/projects/chestxray/geeticka/bert/converted_data/multilabel'

def in_res(filename): return os.path.join(in_dir, filename)
def out_res(filename): return os.path.join(out_dir, filename)

development_or_test = 'development'

train_df = utils.read_dataframe(in_res('train_original.csv'))
test_df = utils.read_dataframe(in_res('test_original.csv'))

new_train_df, new_dev_df = utils.get_new_train_dev_df(train_df, test_df)

# Now that the new and old dataframes have been generated, need to write these into tsv files

if not os.path.exists(out_res('development')):
    os.makedirs(out_res('development'))
if not os.path.exists(out_res('testing')):
    os.makedirs(out_res('testing'))

# Convert edema severity to ordinal encoding
def convert_to_ordinal(severity):
    if severity == 0:
        return '000'
    elif severity == 1:
        return '100'
    elif severity == 2:
        return '110'
    elif severity == 3:
        return '111'
    else:
        raise Exception("Severity can only be between 0 and 3")

# Convert to the df bert expects for the multilabel case
def get_df_bert_multilabel(df):
    data = []
    i = 0
    for index, row in df.iterrows():
        report = extract_report_from_normalized(row['normalized_report'])
        ordinal_label = convert_to_ordinal(row['edema_severity'])
        data.append([i, ordinal_label, 'a', report])
        i += 1
    df_bert = pd.DataFrame(data, columns='id,label,alpha,text'.split(','))
    return df_bert

# Write the newly formed train and dev files (that are taken from the original train data and are only
# to be used for tuning

new_train_df_bert = get_df_bert_multilabel(new_train_df)
new_train_df_bert.to_csv(out_res('development/train.tsv'), sep='\t', index=False, header=False)

new_dev_df_bert = get_df_bert_multilabel(new_dev_df)
new_dev_df_bert.to_csv(out_res('development/dev.tsv'), sep='\t', index=False, header=False)

# Write the original train and test files (that are to be used for reporting)
train_df_bert = get_df_bert_multilabel(train_df)
train_df_bert.to_csv(out_res('testing/train.tsv'), sep='\t', index=False, header=False)

test_df_bert = get_df_bert_multilabel(test_df)
test_df_bert.to_csv(out_res('testing/dev.tsv'), sep='\t', index=False, header=False)
# # we must call it dev for the purposes of the evaluation - that is just the name that the algorithm expects
# # this can probably be changed in the future

