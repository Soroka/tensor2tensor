

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os
import random
import tarfile
import pandas as pd
import requests
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry

import tensorflow as tf

EOS = text_encoder.EOS_ID

# Techniques for data prep from See et al. (2017)
dm_single_close_quote = u"\u2019"  # unicode
dm_double_close_quote = u"\u201d"
# Acceptable ways to end a sentence.
END_TOKENS = [
    u".", u"!", u"?", u"...", u"'", u"`", u"\"", dm_single_close_quote,
    dm_double_close_quote, u")"
]

EOS = text_encoder.EOS_ID

# Techniques for data prep from See et al. (2017)
dm_single_close_quote = u"\u2019"  # unicode
dm_double_close_quote = u"\u201d"
# Acceptable ways to end a sentence.
END_TOKENS = [
    u".", u"!", u"?", u"...", u"'", u"`", u"\"", dm_single_close_quote,
    dm_double_close_quote, u")"
]

summ_finalpath =  'gdrive/My Drive/SemanticScholarAbstractSectionSummaryDataSet/'

all_files = tf.gfile.Glob(summ_finalpath + "*")
                
#to delete later
all_files = all_files[0:2]
                

                
def example_generator(all_files, sum_token):
    # Generate examples

    story_summary_split_token = u" <summary> " if sum_token else " "
    
    for file in all_files:
        print(file)
        dataframe = pd.read_parquet( file , engine = 'fastparquet') 
        for i in range(dataframe.shape[0]):
            summary = dataframe['Summary'].iloc[i]
            source = dataframe['paperSection'].iloc[i]
            yield source + ' ' + story_summary_split_token + ' ' + summary
                
def _story_summary_split(story):
    split_str = u" <summary> "
    split_str_len = len(split_str)
    split_pos = story.find(split_str)
    return story[:split_pos], story[split_pos + split_str_len:]  # story, summary    
                
                
def write_raw_text_to_files(all_files, dataset_split, tmp_dir):

    def write_to_file(all_files, tmp_dir, filename):
    #Write text to files
        with io.open( os.path.join(tmp_dir, filename + ".source"), "w", encoding="utf-8") as fsource:
            with io.open( os.path.join(tmp_dir, filename + ".target"), "w", encoding="utf-8") as fsummary:
                for example in example_generator(all_files, sum_token=True):
                    source, summary = _story_summary_split(example)
                    fsource.write(source + "\n")
                    fsummary.write(summary + "\n")

    if dataset_split == problem.DatasetSplit.TRAIN:
        filename = "summaryData.train"
    elif dataset_split == problem.DatasetSplit.EVAL:
        filename = "summaryData.dev"
    else:
        filename = "summaryData.test"

    tf.logging.info("Writing %s" % filename)
    write_to_file(all_files, tmp_dir, filename)
                
                
@registry.register_problem
class SummarizeScientificSectionsGdrive65k(text_problems.Text2TextProblem):
  #Summarize CNN and Daily Mail articles to their summary highlights.

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        del data_dir
#         all_files, urls_path = _maybe_download_corpora(tmp_dir,
#                                                        problem.DatasetSplit.TRAIN)
        return example_generator(all_files, sum_token=False)

    @property
    def dataset_splits(self):
    #Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 100,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 10,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 10,
        }]

    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
#         all_files, urls_path = _maybe_download_corpora(tmp_dir, dataset_split)
        write_raw_text_to_files(all_files, dataset_split, tmp_dir)
        for example in example_generator(all_files, sum_token=True):
            story, summary = _story_summary_split(example)
            yield {"inputs": story, "targets": summary}
        
    @property
    def approx_vocab_size(self):
    #Approximate vocab size to generate. Only for VocabType.SUBWORD."""
        return 2**16  # ~32k
    #Santosh Edit: changed from 2**15 to 2**16
    
    @property
    def max_subtoken_length(self):
    """Maximum subtoken length when generating vocab.
    SubwordTextEncoder vocabulary building is quadratic-time wrt this variable,
    setting it to None uses the length of the longest token in the corpus.
    Returns:
      an integer or None
    """
        return 50
    
    
