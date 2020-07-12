import spacy
import scispacy
nlp = spacy.load('en_core_sci_md')
#from scispacy.umls_linking import UmlsEntityLinker
import pandas as pd 
from ast import literal_eval

#linker = UmlsEntityLinker()
#nlp.add_pipe(linker)
# There is a problem using the UmlsLinker with this conda environment
# It depends on a package known as nmslib which throws the error
# /data/vision/polina/shared_software/anaconda3-4.3.1/envs/chestxray_joint/lib/python3.6/site-packages/nmslib.cpython-36m-x86_64-linux-gnu.so:
# undefined symbol: _ZTINSt6thread6_StateE
# when I try to import the UmlsEntityLiner
# Would need to rely on workarounds such as https://github.com/aboSamoor/polyglot/issues/78 for this
# This library has other issues like https://github.com/allenai/scispacy/issues/137
# I have opened 2 issues related to this https://github.com/nmslib/nmslib/issues/410 and
# https://github.com/allenai/scispacy/issues/165

# remove any additional whitespace within a line
def remove_whitespace(line):
    return str(" ".join(line.split()).strip())

def list_to_string(sentence):
    return " ".join(sentence)

# normalize digits and remove cases when the same punctuation follows the same one
def normalize_report(row):
    report = row['original_report']
    report_sentences = nlp(report)
    new_report_sentences = []
    for sentence in report_sentences.sents:
        index_to_keep_dict = {} # index: {keep that token or not, replace_with}
        for index in range(0, len(sentence)):
            token = sentence[index]
            if index < len(sentence) - 1:
                next_token = sentence[index + 1]
                if token.is_punct and next_token.is_punct and token.text.strip() == next_token.text.strip():
                    # when it is the same type of punctuation
                    index_to_keep_dict[index] = {'keep': False, 'replace_with': None}
                    continue
            if token.like_num:
                index_to_keep_dict[index] = {'keep': True, 'replace_with': 'NUMBER'}
            else:
                index_to_keep_dict[index] = {'keep': True, 'replace_with': None}
        # generate a new sentence based on this replacement
        new_sentence = []
        for index in range(0, len(sentence)):
            token = sentence[index]
            if not index_to_keep_dict[index]['keep']:
                continue # don't append when there is a double punctuation happening
            if index_to_keep_dict[index]['replace_with'] is not None:
                new_sentence.append(index_to_keep_dict[index]['replace_with'])
                continue
            new_sentence.append(token.text)
        s = list_to_string(new_sentence).strip()
        s = s.replace('DEID', '')
        s = remove_whitespace(s)
        new_report_sentences.append(s)
    return {'sentences': new_report_sentences}

# for pre-processed reports, need to read them differently
# to streamline the reading of the dataframe
def read_dataframe(filepath):
    df = pd.read_csv(filepath, sep='\t')
    def literal_eval_col(row, col_name):
        col = row[col_name]
        col = literal_eval(col)
        return col
    df['metadata'] = df.apply(literal_eval_col, args=('metadata',), axis=1)
    df['normalized_report'] = df.apply(literal_eval_col, args=('normalized_report',), axis=1)
    #df['edema_severity'] = df.apply(literal_eval_col, args=('edema_severity',), axis=1)
    # metadata is a dictionary which is written into the csv format as a string
    # but in order to be treated as a dictionary it needs to be evaluated
    return df

