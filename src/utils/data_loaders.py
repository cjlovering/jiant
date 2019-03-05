"""
Functions having to do with loading data from output of
files downloaded in scripts/download_data_glue.py

"""
from .tokenizers import get_tokenizer
import codecs
import pandas as pd
import csv
import numpy as np
from allennlp.data import vocabulary

BERT_CLS_TOK, BERT_SEP_TOK = "[CLS]", "[SEP]"
SOS_TOK, EOS_TOK = "<SOS>", "<EOS>"

def load_tsv(
        tokenizer_name,
        data_file,
        max_seq_len,
        label_idx=2,
        s1_idx=0,
        s2_idx=1,
        label_fn=None,
        skip_rows=0,
        return_indices=False,
        delimiter='\t',
        filter_idx=None,
        has_labels=True,
        filter_value=None):
    '''
    Load a tsv.
    To load only rows that have a certain value for a certain column,
    like genre in MNLI, set filter_idx and filter_value (for example,
    for mnli-fiction  we want columns where genre == 'fiction' ).
    Args:
        s1_idx; int
        s2_idx: int
        targ_idx: int
        filter_idx: int this is the index that we want to filter from
        filter_value: string the value in which we want filter_idx to be equal to
        return_indices: bool that describes if you need to return indices (for purposes of matching)
        label_fn is a function that expects a row and outputs the label
    Returns:
        List of first and second sentences, labels, and if applicable indices
    '''
    # TODO(Yada): Instead of index integers, adjust this to pass ins column names
    sent1s, sent2s, labels = pd.Series(), pd.Series(), pd.Series()
    # This reads the data file given the delimiter, skipping over any rows (usually header row)
    rows = pd.read_csv(data_file, \
                        sep=delimiter, \
                        error_bad_lines=False, \
                        header=None, \
                        skiprows=skip_rows, \
                        quoting=csv.QUOTE_NONE,\
                        encoding='utf-8')
    if filter_idx:
        rows = rows[rows[filter_idx] == filter_value]
    # Filter for sentence1s that are of length 0
    # Filter if row[targ_idx] is nan
    mask = (rows[s1_idx].str.len() > 0)
    if has_labels:
        mask = mask & (~rows[label_idx].isnull())
    rows = rows.loc[mask]
    sent1s = rows[s1_idx].apply(lambda x: process_sentence(tokenizer_name, x, max_seq_len))
    if s2_idx:
        sent2s = rows[s2_idx].apply(lambda x: process_sentence(tokenizer_name, x, max_seq_len))

    if has_labels:
        if label_fn is None:
            labels = rows[label_idx]
        else:
            labels = rows[label_idx].apply(lambda x: label_fn(x))
    else:
        # If dataset doesn't have labels, for example for test set, then mock labels
        labels = np.zeros(len(rows), dtype=int)
    if return_indices:
        idxs = rows.index.tolist()
        # Get indices of the remaining rows after filtering
        return sent1s.tolist(), sent2s.tolist(), labels.tolist(), idxs
    else:
        return sent1s.tolist(), sent2s.tolist(), labels.tolist()

def load_diagnostic_tsv(
        tokenizer_name,
        data_file,
        max_seq_len,
        label_col,
        s1_col="",
        s2_col="",
        label_fn=None,
        skip_rows=0,
        delimiter='\t'):
    '''Load a tsv and  indexes the columns from the diagnostic tsv.
        This is only used for MNLI-diagnostic right now.
    Args:
        data_file: string
        max_seq_len: int
        s1_col: string
        s2_col: string
        label_col: string
        label_fn: function
        skip_rows: list of ints
        delimiter: string
    Returns:
        A dictionary of the necessary indexed fields, the tokenized sent1 and sent2
        and indices
        Note: If a field in a particular row in the dataset is empty, we return []
        for that field for that row, otherwise we return an array of ints (indices)
        Else, we return an array of indices
    '''
    # TODO: Abstract indexing layer from this function so that MNLI-diagnostic calls load_tsv
    assert len(s1_col) > 0 and len(label_col) > 0, "Make sure you passed in column names for sentence 1 and labels"
    sent1s, sent2s, targs, idxs, lex_sem, pr_ar_str, logic, knowledge = pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series(), pd.Series()
    rows = pd.read_csv(data_file, \
                        sep=delimiter, \
                        error_bad_lines=False, \
                        quoting=csv.QUOTE_NONE,\
                        encoding='utf-8')
    rows = rows.fillna('')
    def targs_to_idx(col_name):
        # This function builds the index to vocab (and its inverse) mapping
        values = set(rows[col_name].values)
        vocab = vocabulary.Vocabulary(counter=None)
        for value in values:
            vocab.add_token_to_namespace(value, col_name)
        idx_to_word = vocab.get_index_to_token_vocabulary(col_name)
        word_to_idx = vocab.get_token_to_index_vocabulary(col_name)
        rows[col_name] = rows[col_name].apply(lambda x: [word_to_idx[x]] if x != '' else [])
        return word_to_idx, idx_to_word, rows[col_name]

    sent1s = rows[s1_col].apply(lambda x: process_sentence(tokenizer_name, x, max_seq_len))
    sent2s = rows[s2_col].apply(lambda x: process_sentence(tokenizer_name, x, max_seq_len))
    labels = rows[label_col].apply(lambda x: label_fn(x))
    # Build indices for field attributes
    lex_sem_to_ix_dic, ix_to_lex_sem_dic, lex_sem = targs_to_idx("Lexical Semantics")
    pr_ar_str_to_ix_di, ix_to_pr_ar_str_dic, pr_ar_str = targs_to_idx("Predicate-Argument Structure")
    logic_to_ix_dic, ix_to_logic_dic, logic = targs_to_idx("Logic")
    knowledge_to_ix_dic, ix_to_knowledge_dic, knowledge = targs_to_idx("Knowledge")
    idxs = rows.index

    return {'sents1': sent1s.tolist(),
            'sents2': sent2s.tolist(),
            'targs': labels.tolist(),
            'idxs': idxs.tolist(),
            'lex_sem': lex_sem.tolist(),
            'pr_ar_str': pr_ar_str.tolist(),
            'logic': logic.tolist(),
            'knowledge': knowledge.tolist(),
            'ix_to_lex_sem_dic': ix_to_lex_sem_dic,
            'ix_to_pr_ar_str_dic': ix_to_pr_ar_str_dic,
            'ix_to_logic_dic': ix_to_logic_dic,
            'ix_to_knowledge_dic': ix_to_knowledge_dic
            }

def process_sentence(tokenizer_name, sent, max_seq_len):
    '''process a sentence '''
    max_seq_len -= 2
    assert max_seq_len > 0, "Max sequence length should be at least 2!"
    tokenizer = get_tokenizer(tokenizer_name)
    if tokenizer_name.startswith("bert-"):
        sos_tok, eos_tok = BERT_SEP_TOK, BERT_CLS_TOK
    else:
         sos_tok, eos_tok = SOS_TOK, EOS_TOK
    if isinstance(sent, str):
        return [sos_tok] + tokenizer.tokenize(sent)[:max_seq_len] + [eos_tok]
    elif isinstance(sent, list):
        assert isinstance(sent[0], str), "Invalid sentence found!"
        return [sos_tok] + sent[:max_seq_len] + [eos_tok]