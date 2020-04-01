# -*-coding:utf-8-*-
#! /usr/bin/env python
# https://github.com/ffancellu/NegNN

from collections import Counter
from itertools import chain
from argparse import ArgumentParser

from conll2obj import Data

import os
import pickle
import numpy as np
import sys
import codecs


def load_data(path, scope, event, lang):
    # read data,get sentences as list of lists
    raw_data = Data(path)

    # get all strings
    sents, tags, tags_uni, labels, cues, scopes, lengths = data2sents(
        [raw_data], event, scope, lang
    )

    # build vocabularies
    voc, voc_inv = build_vocab(sents, tags, tags_uni, labels, lengths)

    # transform the tokens into integer indices
    tags_idxs, tags_uni_idxs, cues_idxs, scopes_idxs, labels_idxs = build_input_data(
        voc, sents, tags, tags_uni, cues, scopes, labels
    )

    data_generator = package(
        sents,
        tags,
        tags_uni,
        labels,
        cues,
        scopes,
        tags_idxs,
        tags_uni_idxs,
        cues_idxs,
        scopes_idxs,
        labels_idxs,
    )
    data = list(data_generator)

    return data


def package(
    sents,
    tags,
    tags_uni,
    labels,
    cues,
    scopes,
    tags_idxs,
    tags_uni_idxs,
    cues_idxs,
    scopes_idxs,
    labels_idxs,
):
    for (
        sent,
        tag,
        tag_uni,
        label,
        cue,
        scope,
        tags_idx,
        tags_uni_idx,
        cues_idx,
        scopes_idx,
        labels_idx,
    ) in zip(
        sents,
        tags,
        tags_uni,
        labels,
        cues,
        scopes,
        tags_idxs,
        tags_uni_idxs,
        cues_idxs,
        scopes_idxs,
        labels_idxs,
    ):
        yield {
            "sent": sent,
            "tag": tag,
            "tag_uni": tag_uni,
            "label": label,
            "cue": cue,
            "scope": scope,
            "tags_idx": tags_idx.tolist(),
            "tags_uni_idx": tags_uni_idx.tolist(),
            "cues_idx": cues_idx.tolist(),
            "scopes_idx": scopes_idx.tolist(),
            "labels_idx": labels_idx.tolist(),
        }


def build_vocab(sents, tags, tags_uni, labels, lengths):
    def token2idx(cnt):
        return dict([(w, i) for i, w in enumerate(cnt.keys())])

    w2idxs = token2idx(Counter(chain(*sents)))
    # add <UNK> token
    w2idxs["<UNK>"] = max(w2idxs.values()) + 1
    t2idxs = token2idx(Counter(chain(*tags)))
    tuni2idxs = token2idx(Counter(chain(*tags_uni)))
    y2idxs = {"I": 0, "O": 1, "E": 2}

    voc, voc_inv = {}, {}
    voc["w2idxs"], voc_inv["idxs2w"] = w2idxs, {i: x for x, i in w2idxs.items()}
    voc["y2idxs"], voc_inv["idxs2y"] = y2idxs, {i: x for x, i in y2idxs.items()}
    voc["t2idxs"], voc_inv["idxs2t"] = t2idxs, {i: x for x, i in t2idxs.items()}
    voc["tuni2idxs"], voc_inv["idxs2tuni"] = (tuni2idxs, {x: i for x, i in tuni2idxs.items()})

    return voc, voc_inv


def build_input_data(voc, sents, tags, tags_uni, cues, scopes, labels):

    tags_idxs = [
        np.array([voc["t2idxs"][t] for t in tag_sent], dtype=np.int32) for tag_sent in tags
    ]
    tags_uni_idxs = [
        np.array([voc["tuni2idxs"][tu] for tu in tag_sent_uni], dtype=np.int32)
        for tag_sent_uni in tags_uni
    ]
    y_idxs = [np.array([voc["y2idxs"][y] for y in y_array], dtype=np.int32) for y_array in labels]
    cues_idxs = [
        np.array([1 if c == "CUE" else 0 for c in c_array], dtype=np.int32) for c_array in cues
    ]
    scope_idxs = [
        np.array([1 if s == "S" else 0 for s in s_array], dtype=np.int32) for s_array in scopes
    ]

    return tags_idxs, tags_uni_idxs, cues_idxs, scope_idxs, y_idxs


def package_data_train_dev(
    sent_ind_x, tag_ind_x, tag_uni_ind_x, sent_ind_y, cues_idxs, scopes_idxs, voc, voc_inv, lengths
):

    # vectors of words
    train_x, dev_x = (sent_ind_x[: lengths[0]], sent_ind_x[lengths[0] : lengths[0] + lengths[1]])

    # vectors of POS tags
    train_tag_x, dev_tag_x = (
        tag_ind_x[: lengths[0]],
        tag_ind_x[lengths[0] : lengths[0] + lengths[1]],
    )

    # vectors of uni POS tags
    train_tag_uni_x, dev_tag_uni_x = (
        tag_uni_ind_x[: lengths[0]],
        tag_uni_ind_x[lengths[0] : lengths[0] + lengths[1]],
    )

    # vectors of y labels
    train_y, dev_y = (sent_ind_y[: lengths[0]], sent_ind_y[lengths[0] : lengths[0] + lengths[1]])

    # vectors of cue info
    train_cue_info, dev_cue_info = (
        cues_idxs[: lengths[0]],
        cues_idxs[lengths[0] : lengths[0] + lengths[1]],
    )

    # vectors of scope info
    train_scope_info, dev_scope_info = (
        scopes_idxs[: lengths[0]],
        scopes_idxs[lengths[0] : lengths[0] + lengths[1]],
    )

    train_set = [train_x, train_tag_x, train_tag_uni_x, train_y, train_cue_info, train_scope_info]
    dev_set = [dev_x, dev_tag_x, dev_tag_uni_x, dev_y, dev_cue_info, dev_scope_info]

    return [train_set, dev_set, voc, voc_inv]


# try different data formats
def data2sents(sets, look_event, look_scope, lang):
    def get_uni_mapping(lang):
        mapping = {}
        f = codecs.open("./data/uni_pos_map/%s.txt" % lang, "rb", "utf8").readlines()
        for line in f:
            spl = line.strip().split("\t")
            _pos = spl[0].split("|")[0]
            mapping.update({_pos: spl[1]})
        return mapping

    def segment(word, is_cue):
        _prefix_one = ["a"]
        _prefix_two = ["ab", "un", "im", "in", "ir", "il"]
        _prefix_three = ["dis", "non"]
        _suffix = ["less", "lessness", "lessly"]

        which_suff = [word.endswith(x) for x in _suffix]

        if is_cue:
            if word.lower()[:2] in _prefix_two and len(word) > 4:
                return ([word[:2] + "-", word[2:]], 1)
            elif word.lower()[:1] in _prefix_one and len(word) > 4:
                return ([word[:1] + "-", word[1:]], 1)
            elif word.lower()[:3] in _prefix_three and len(word) > 4:
                return ([word[:3] + "-", word[3:]], 1)
            elif True in which_suff and len(word) > 4:
                idx = [i for i, v in enumerate(which_suff) if v][0]
                suff_cue = _suffix[idx]
                return ([word[: -len(suff_cue)], "-" + suff_cue], 0)
            else:
                return ([word], None)

        else:
            return ([word], None)

    def assign_tag(is_event, is_scope, look_event, look_scope):
        if is_event and look_event:
            return "E"
        elif is_scope and look_scope:
            return "I"
        else:
            return "O"

    sents = []
    tag_sents = []
    ys = []
    lengths = []

    cues_one_hot = []
    scopes_one_hot = []

    for d in sets:
        length = 0
        for s_idx, s in enumerate(d):
            all_cues = [
                i for i in range(len(s)) if filter(lambda x: x.cue != None, s[i].annotations) != []
            ]
            if len(s[0].annotations) > 0:
                for curr_ann in range(len(s[0].annotations)):
                    cues_idxs = [
                        i[0]
                        for i in filter(
                            lambda x: x[1] != None,
                            [(i, s[i].annotations[curr_ann].cue) for i in range(len(s))],
                        )
                    ]
                    event_idxs = [
                        i[0]
                        for i in filter(
                            lambda x: x[1] != None,
                            [(i, s[i].annotations[curr_ann].event) for i in range(len(s))],
                        )
                    ]
                    scope_idxs = [
                        i[0]
                        for i in filter(
                            lambda x: x[1] != None,
                            [(i, s[i].annotations[curr_ann].scope) for i in range(len(s))],
                        )
                    ]

                    sent = []
                    tag_sent = []
                    y = []

                    cue_one_hot = []
                    scope_one_hot = []

                    for t_idx, t in enumerate(s):
                        word, tag = t.word, t.pos
                        word_spl, word_idx = segment(word, t_idx in all_cues)
                        if len(word_spl) == 1:
                            _y = assign_tag(
                                t_idx in event_idxs, t_idx in scope_idxs, look_event, look_scope
                            )
                            c_info = ["NOTCUE"] if t_idx not in cues_idxs else ["CUE"]
                            s_info = ["S"] if t_idx in scope_idxs else ["NS"]
                            tag_info = [tag]

                        elif len(word_spl) == 2:
                            _y_word = assign_tag(
                                t_idx in event_idxs, t_idx in scope_idxs, look_event, look_scope
                            )
                            if t_idx in cues_idxs:
                                _y = [_y_word, "O"] if word_idx == 0 else ["O", _y_word]
                                c_info = ["NOTCUE", "CUE"] if word_idx == 0 else ["CUE", "NOTCUE"]
                                s_info = ["S", "NS"] if word_idx == 0 else ["NS", "S"]
                            else:
                                _y = [_y_word, _y_word]
                                c_info = ["NOTCUE", "NOTCUE"]
                                s_info = ["S", "S"] if t_idx in scope_idxs else ["NS", "NS"]
                            tag_info = [tag, "AFF"] if word_idx == 0 else ["AFF", tag]
                        # add the word(s) to the sentence list
                        sent.extend(word_spl)
                        # add the POS tag(s) to the TAG sentence list
                        tag_sent.extend(tag_info)
                        # add the _y for the word
                        y.extend(_y)
                        # extend the cue hot vector
                        cue_one_hot.extend(c_info)
                        # extend the scope hot vector
                        scope_one_hot.extend(s_info)

                    sents.append(sent)
                    tag_sents.append(tag_sent)
                    ys.append(y)
                    cues_one_hot.append(cue_one_hot)
                    scopes_one_hot.append(scope_one_hot)
                    length += 1

        lengths.append(length)
    # make normal POS tag into uni POS tags
    pos2uni = get_uni_mapping(lang)
    tag_uni_sents = [[pos2uni[t] for t in _s] for _s in tag_sents]

    return sents, tag_sents, tag_uni_sents, ys, cues_one_hot, scopes_one_hot, lengths
