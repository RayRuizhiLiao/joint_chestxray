'''
Author: Geeticka Chauhan
This file consists of the utils related to the language modeling
We will be processing all the reports here 
'''
import pandas as pd
from ast import literal_eval
import csv
import re
import numpy as np
import random
import os
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import glob
import re
from pyexpat import ExpatError

# read the reports dataframe 
def get_reports_df(in_dir):
    total_files_to_read = glob.glob(in_dir + '*.txt')
    print('Total files to read:' , len(total_files_to_read) , ' from dir: ' , in_dir)
    
    data = []
    for filepath in tqdm(total_files_to_read):
        filename = os.path.basename(filepath)
        try:
            report_dictionary = get_report_dictionary(filepath)
            if not report_dictionary: # when the report is empty
                continue
            data.append([filename, report_dictionary])
        except:
            raise Exception('Following file threw an error %s'%filename)
    df = pd.DataFrame(data, columns='filename,report'.split(','))
    return df

'''
Helper functions for parsing the text in the report 
'''
# recursively fix all DEID tokens
def fix_DEID(line):
    m = re.search('([^_]___[^_])|(^___[^_])|([^_]___$)', line) 
    '''
    eg: 
    re.search('[^_]___[^_]', ' .___a  _______') # match 1 or more underscores
    >> <_sre.SRE_Match object; span=(1, 6), match='.___a'>
    '''
    if m:
        span = m.span()
        return fix_DEID(line[:span[0]+1] + ' DEID ' + line[span[1]-1:])
    else:
        return line

# check if the dictionary contains that section already
# if yes then add the value as a list rather than replacing the value
# pass by val or reference??? check this
def append_to_dict(filename, dictionary, section, value):
    if section in dictionary:
        #print('File %s involves appending to the dictionary value list'%filename)
        cur_val = dictionary[section]
        if type(dictionary[section]) is list:
            cur_val.append(value)
            dictionary[section] = cur_val
        else:
            dictionary[section] = [cur_val, value]
    else:
        dictionary[section] = value
    return dictionary

# given a path to a report txt, get the report dictionary
def get_report_dictionary(filepath):
    with open(filepath, 'r') as text_file:
        filename = os.path.basename(filepath)
        report_dictionary = {} # you write directly to report dictionary if not in final report section yet
        final_report_dictionary = {} # write to the final report section if you have reached final report now
        current_section = '' 
        previous_section = ''
        current_section_lines = '' # when switching sections we will want to write into the report dictionary
        final_report = False # have we reached final report section yet? 
        # A report will look like sections : {'WET READ':..., 'FINAL REPORT':....}

        for line in text_file.readlines():
            line = line.strip() # I just removed a \n at the end of each line.
            old_line = line
            # look for DEID tokens
            line = fix_DEID(line)
            line = remove_whitespace(line)
    #        if 'DEID' in line:
     #           print('%s file contains DEID token and line is now %s from %s'%(filename, line, old_line))
            line = line.replace('_', '') # this is the deid token; in the future could replace 3 underscores
            # with the deid token and then replace more underscores with nothing
            #print(line)
            if not line:
                continue
            '''
            Figure out whether we are in the final report part of the report
            '''
            # various checks to see if we are in the final report section
            if line.startswith('FINAL REPORT'):
                final_report = True
                # but before this final report section, need to see if a section had been written and need to 
                # put that in 
                # TODO: fix this 
                if current_section: #if prev section is not empty we just started a new one
                    # but we also need to know when we are at the end of file, so we can insert the last
                    # section as well
                    current_section = "_".join(current_section.lower().split())
                    current_section_lines = remove_whitespace(current_section_lines)
                    # write a method to test if the current report dictionary contains 
                    report_dictionary = append_to_dict(filename, report_dictionary, current_section,
                            current_section_lines)
                    previous_section = current_section
                    current_section = "" # need to reset everything because when the next section is seen, 
                    # we don't want to append this section into it. 
                    current_section_lines = ""
            if line == 'FINAL REPORT':
                continue
            elif line.startswith('FINAL REPORT'): # in this case there is some text right after final rep
                print('report %s has the following text right after FINAL REPORT: %s'%(filename, line))
                #current_section = 'no_section'
                #m = re.search('(?<=FINAL REPORT:).*', line)
                #line_following_section = m.group(0).strip()
                #if line_following_section: current_section_lines += line_following_section
                # ideally this is not happening with any report but handle it if it does
                continue
            if final_report == True: dict_to_append = final_report_dictionary
            else: dict_to_append = report_dictionary

            # Now that we have decided the report dict to append to
            # we can figure out which section we are in at the moment

            '''
            Let's decide which section of the report we are present in
            '''
            # Are we at the beginning of a section?
            # If we are at the beginning of a section, we need to write the previous section into 
            # the correct dictionary
            # only look at the regular expressions that are
            # lines starting with non numbers followed by the first :
            #if re.match('[a-zA-Z]+:.*', line): # means we began a section
            # if re.match('^[a-zA-z]+(([\s]{1,2}|/)[a-zA-Z]+)*:.*', line)
            # Below will miss cases where FINDING and RECOMMENDATION are not 
            # present in the beginning of the text re.search('FINDING[(|S|)]*:', line)
            trailing = '(\.|(DEID))[\s]*%s[(|S|)]*:' # when . IMPRESSION(S) type of thing appears
            if re.match('^[a-zA-z]+(([\s]{1,2}|/)[a-zA-Z]+)*:.*', line) or ((re.search('^FINDING', line) or \
                    re.search('^IMPRESSION', line) or  re.search('^CONCLUSION', line) or
                    re.search('^RECOMMENDATION', line) or re.search(trailing%'FINDING', line) or \
                            re.search(trailing%'IMPRESSION', line) or re.search(trailing%'CONCLUSION', line) or \
                            re.search(trailing%'RECOMMENDATION', line)) and final_report): # space is allowed between two words
                # make sure that you check for the IMRESSION: FINDINGS: cases 
                # this should only be applied to ['s54475003.txt', 's56509801.txt', 's51485529.txt',
                # 's51017605.txt']
                # or slash is allowed but no numbers are allowed
                # below will miss cases like in s56653797.txt where CHEST: follows right after FINDINGS/IMPR..

                ''' 
                Realm of previous section not having been written into dict yet
                '''
                if re.search(trailing%'FINDING', line) or re.search(trailing%'IMPRESSION', line) or \
                        re.search(trailing%'CONCLUSION', line) or re.search(trailing%'RECOMMENDATION', line):
                            next_sec_pattern = '[A-Z()]*:'
                            next_sec = re.search(next_sec_pattern, line).group(0) # the object that matches
                            current_section_lines += ' ' + re.split(next_sec_pattern, line)[0] # what lies before
                # include a case where there are findings and impressions but there is a new section following
                # which contains too many spaces (more than 3)
                imp_sec_empty = False
                if ('finding' in current_section.lower() or 'impression' in current_section.lower()) and \
                        current_section_lines == '':
                        imp_sec_empty = True

                if current_section: #if prev section is not empty we just started a new one
                    # but we also need to know when we are at the end of file, so we can insert the last
                    # section as well
                    current_section = "_".join(current_section.lower().split())
                    current_section_lines = remove_whitespace(current_section_lines)
                    dict_to_append = append_to_dict(filename, dict_to_append, current_section,
                            current_section_lines)
                    previous_section = current_section
                current_section_lines = ''
                
                '''
                Now that we have prev section in the dictionary, we need to grab the current section
                '''
                # match those that have alphabets, spaces, / before quote and not those cases where 
                # we have IMPRESSION: FINDINGS
                if re.match('^[a-zA-z]+(([\s]{1,2}|/)[a-zA-Z]+)*:.*', line) and not re.search('^IMPRESSION:[\s]*FINDINGS:.*', line):
                    current_section = line.split(':')[0]
                    if imp_sec_empty and (len(current_section.split()) - 1) > 3:
                        print('File %s has importance sections empty and following section with too many space'%filename)
                        current_section = previous_section
                        current_section_lines += line
                        continue
                elif re.search(trailing%'FINDING', line) or re.search(trailing%'IMPRESSION', line) or \
                        re.search(trailing%'CONCLUSION', line) or re.search(trailing%'RECOMMENDATION', line):
                            current_section = next_sec
                else: # this will be greedy and grab whatever text is before FINDING or IMPRESSION or 
                    # CONCLUSION....
                    words_to_look_for = ['FINDING', 'IMPRESSION', 'CONCLUSION', 'RECOMMENDATION']
                    words_found = []
                    for word in words_to_look_for:
                        if word in line:
                            words_found.append(word)
                        if len(words_found) > 1:
                            print('Line %s in file %s has both section names %s'%(line, filename, words_found))
                    if len(words_found) == 1:
                        word_found = words_found[0]
                    elif len(words_found) > 1: # basically look for the indicator word that appears latest in line
                        span_end = 0
                        for word in words_found:
                            curr_span = re.match('^.*%s'%word, line).span()[1]
                            if curr_span > span_end: # can't be equal as span should at least be 1, word_found
                                # must therefore never be None
                                span_end = curr_span
                                word_found = word
                    splitted = line.split(word_found)
                    cur_sec_loc = re.match('^.*%s'%word_found, line).span()[1]
                    current_section = line[:cur_sec_loc]
                    if splitted[1] and not splitted[1][0].isspace() and not splitted[1][0] == ':': # if the next portion of the line doesnt begin with space
                        current_section_spiller = splitted[1].split()[0]
                        current_section += current_section_spiller
                        # below is a hack but splitting it by : does not work well for IMPRESSION: FINDINGS:
                        # cases
                        if not re.search('^IMPRESSION:[\s]*FINDINGS:.*', current_section):
                            current_section = current_section.split(':')[0] # sometimes : might come in anyway
                    # look for the first space in line.split[1] and merge that with current_section
                current_sec_reg = current_section.replace('(', '\(')
                current_sec_reg = current_sec_reg.replace(')', '\)')
                #current_sec_reg = current_sec_reg.replace('\\', '\\\\')
                #current_sec_reg = current_sec_reg.replace('/', '\/')
                m = re.search('(?<=%s:).*'%current_sec_reg, line) # assumes that : follows
                if not m: # in the cases where CONCLUSION matched but no : present
                    m = re.search('(?<=%s).*'%current_sec_reg, line)
                line_following_section = m.group(0).strip()
                if line_following_section: current_section_lines += line_following_section
                '''
                now that all the work related to finding the section names exactly within line is done, 
                there are some missed cases when : is present in the key like in the case of 
                IMPRESSION: FINDINGS: let's remove that 
                or cases like FINDINGS/ IMPRESSION in which case / becomes part of the key. 
                Remove that too
                '''
                current_section = ''.join(current_section.split(':'))
                current_section = ''.join(current_section.split('/'))

                continue # already wrote all the text that was present in this line

            # edge case of where no sections have started but text was written
            # we know text was written because if line had been empty we would not have gotten this far
            elif not current_section:
                current_section = 'no_section'

            '''
            We are now in the middle of a section and can append to the lines 
            '''
            current_section_lines += " " + line  #add a space for now, and can later remove additional spaces
    
    if not current_section: # if current section is empty but the lines are 
        # not then we want to append the lines to the previous section
        print('The file %s is empty'%filename)
        return {'final_report':  ''}
    # TODO geeticka: below is more complicated and need to 
    # make sure previous_section is updated correctly
    # and that the current section lines just get appended to what was already
# present in the dictionary for the previous section
# at the moment things are working but would be good to improve the code 
    # in this sense
    if not current_section and not current_section_lines:
        current_section = previous_section
    current_section = "_".join(current_section.lower().split())
    current_section_lines = remove_whitespace(current_section_lines)
    dict_to_append = append_to_dict(filename, dict_to_append, current_section, current_section_lines)
    report_dictionary['final_report'] = final_report_dictionary
        
    return report_dictionary


'''
Below are just helper functions
'''
# to streamline the writing of the dataframe
def write_dataframe(df, filepath):
    df.to_csv(filepath, sep='\t', encoding='utf-8', index=False)

# to streamline the reading of the dataframe
def read_dataframe(filepath):
    df = pd.read_csv(filepath, sep='\t')
    def literal_eval_col(row, col_name):
        col = row[col_name]
        col = literal_eval(col)
        return col
    df['report'] = df.apply(literal_eval_col, args=('report',), axis=1)
    #df['edema_severity'] = df.apply(literal_eval_col, args=('edema_severity',), axis=1)
    # metadata is a dictionary which is written into the csv format as a string
    # but in order to be treated as a dictionary it needs to be evaluated
    return df

# The goal here is to make sure that the df that is written into memory is the same one that is read
def check_equality_of_written_and_read_df(df, df_copy):
    bool_equality = df.equals(df_copy)
    # to double check, we want to check with every column
    bool_every_column = True
    for idx in range(len(df)):
        row1 = df.iloc[idx]
        row2 = df_copy.iloc[idx]
        # for any column names (grab the column names), then compare for all of them and print the 
        # column name where they differ
        if not np.array_equal(row1.index, row2.index):
            print("The two dataframes must have identical columns with their order!")
            return
        columns = row1.index
        for column_name in columns:
            if row1[column_name] != row2[column_name]:
                print("The dataframes differ in column: ", column_name)
                bool_every_column = False
                return bool_equality, bool_every_column
    return bool_equality, bool_every_column

# remove any additional whitespace within a line
def remove_whitespace(line):
    return str(" ".join(line.split()).strip())
