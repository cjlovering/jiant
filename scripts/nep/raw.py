# -*-coding:utf-8-*-
#! /usr/bin/env python
# https://github.com/ffancellu/NegNN

import argparse
import codecs
import json
import os
import sys
import time

import pandas as pd

import processor


def main():

    # Parameters
    # ==================================================
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_sent_length", default=100, help="Maximum sentence length for padding (default:100)"
    )
    parser.add_argument("--num_classes", default=2, help="Number of y classes (default:2)")

    # Training parameters
    parser.add_argument(
        "--scope_detection",
        default=True,
        help="True if the task is scope detection or joined scope/event detection",
    )
    parser.add_argument(
        "--event_detection",
        default=False,
        help="True is the task is event detection or joined scope/event detection",
    )
    parser.add_argument(
        "--POS_emb", default=0, help="0: no POS embeddings; 1: normal POS; 2: universal POS"
    )
    parser.add_argument(
        "--training_lang", default="en", help="Language of the tranining data (default: en)"
    )
    FLAGS = parser.parse_args()

    # Data Preparation
    # ==================================================

    fn_dev = os.path.abspath("./data/dev/sherlock_dev.txt")
    dev_data = processor.load_data(
        fn_dev, FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang
    )
    if not os.path.isdir("./raw/"):
        os.mkdir("./raw/")
    pd.DataFrame(dev_data).to_csv(
        "./raw/dev.tsv",
        columns=["sent", "cues_idx", "label", "labels_idx", "scopes_idx", "scope"],
        sep="\t",
        index=False,
    )

    fn_training = os.path.abspath("./data/training/sherlock_train.txt")
    train_data = processor.load_data(
        fn_training, FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang
    )
    pd.DataFrame(train_data).to_csv(
        "./raw/training.tsv",
        columns=["sent", "cues_idx", "label", "labels_idx", "scopes_idx", "scope"],
        sep="\t",
        index=False,
    )

    tests = [
        "sherlock_cardboard.txt",
        "sherlock_circle.txt",
        "simple_wiki/full/unseen_full.conll",
        "simple_wiki/full/lexical_full.conll",
        "simple_wiki/full/mw_full.conll",
        "simple_wiki/full/prefixal_full.conll",
        "simple_wiki/full/simple_full.conll",
        "simple_wiki/full/suffixal_full.conll",
        "simple_wiki/full/unseen_full.conll",
    ]
    for test_file in tests:
        fn_test = os.path.abspath(f"./data/test/{test_file}")
        test_data = processor.load_data(
            fn_test, FLAGS.scope_detection, FLAGS.event_detection, FLAGS.training_lang
        )
        test_name = test_file.split("/")[-1].split(".")[0]
        pd.DataFrame(test_data).to_csv(f"./raw/{test_name}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
