# https://github.com/ffancellu/NegNN

import ast
import itertools
import pandas as pd
from collections import Counter
from nltk import Tree
import os

# We use the allennlp parser (I liked the outputs I sampled.)
# https://demo.allennlp.org/constituency-parsing/MTU5NjQxOQ==
from jiant.utils.data_loaders import tokenize_and_truncate
from allennlp.predictors.predictor import Predictor

predictor = Predictor.from_path(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz"
)


def nli():
    files = [
        "dev",
        "training",
        "sherlock_cardboard",
        "sherlock_circle",
        "unseen_full",
        "lexical_full",
        "mw_full",
        "prefixal_full",
        "simple_full",
        "suffixal_full",
        "unseen_full",
    ]
    if not os.path.isdir("./nli/"):
        os.mkdir("./nli/")
    out = []
    for f in files:
        df = pipeline(f)
        out.append(df)
        df.to_csv(
            f"./nli/{f}.tsv",
            index=False,
            sep="\t",
            columns=["sent_1", "sent_2", "label", "case", "cue", "common_cue"],
        )
    out_df = pd.concat(out)
    out_df.to_csv(
        f"./nli/all.tsv",
        index=False,
        sep="\t",
        columns=["sent_1", "sent_2", "label", "case", "cue", "common_cue"],
    )


def random():
    files = [
        "dev",
        "training",
        "sherlock_cardboard",
        "sherlock_circle",
        "unseen_full",
        "lexical_full",
        "mw_full",
        "prefixal_full",
        "simple_full",
        "suffixal_full",
        "unseen_full",
    ]
    out = []
    for f in files:
        df = pipeline_random(f)
        out.append(df)
        df.to_csv(
            f"./random/{f}.tsv",
            index=False,
            sep="\t",
            columns=[
                "label",
                "sent_1",
                "sent_2",
                "positive_1",
                "positive_2",
                "subsequence_1",
                "subsequence_2",
            ],
        )
    out_df = pd.concat(out)
    out_df.to_csv(
        f"./random/all.tsv",
        index=False,
        sep="\t",
        columns=[
            "label",
            "sent_1",
            "sent_2",
            "positive_1",
            "positive_2",
            "subsequence_1",
            "subsequence_2",
        ],
    )


def pipeline(name: str):
    # setup data
    data = pd.read_csv(f"./raw/{name}.tsv", sep="\t")
    # we replace n't with not.
    data.sent = data.sent.apply(lambda x: x.replace("n't", "not"))
    data.sent = data.sent.apply(lambda x: " ".join(["can" if w == "ca" else w for w in x.split()]))
    data["cues"] = (
        data.apply(get_cue, axis=1)
        # lower the cues
        .apply(lambda x: x.lower() if isinstance(x, str) else x)
        # handle cases where the cues are multiple words
        .apply(lambda x: "-".join(x) if isinstance(x, list) else x)
    )

    # add additional fields.
    data["scope"] = data.apply(get_scope, axis=1)
    data["full_scope"] = data.apply(get_full_scope, axis=1)

    # filter cues to a smaller subset.
    cue_filter = set(["not", "no", "never", "nor"])
    data["common_cue"] = data.cues.apply(lambda c: c in cue_filter)

    # filter scopes that area really short.
    df = data[data.scope.apply(len) > 10]

    # generate pairs for entailment classification.
    records = df.to_dict(orient="records")
    df_nli = pd.DataFrame(list(itertools.chain.from_iterable(map(get_nli, records))))

    # filter hypotheses s.t. they are (more) well-formed.
    df_nli = df_nli[df_nli.sent_2.apply(filter_nli)]

    # fix outputs (de-split words like im-possible).
    records = df_nli.to_dict(orient="records")
    df = pd.DataFrame(map(fix_nli, records))
    df = df.drop_duplicates(subset=["sent_1", "sent_2"], ignore_index=True)

    # match outputs for jiant
    # Filter for sentence1s that are of length 0
    # Filter if row[targ_idx] is nan
    mask = df.sent_1.str.len() > 0 & (df.sent_2.str.len() > 0) & rows.label.notnull()
    df = df[mask]
    df.sent_1 = df.sent_1.apply(lambda x: tokenize_and_truncate(tokenizer_name, x, max_seq_len))
    df.sent_2 = df.sent_2.apply(lambda x: tokenize_and_truncate(tokenizer_name, x, max_seq_len))

    return df


def pipeline_random(name: str):
    # setup data
    data = pd.read_csv(f"./raw/{name}.tsv", sep="\t")

    # we replace n't with not.
    data.sent = data.sent.apply(lambda x: x.replace("n't", "not"))
    data.sent = data.sent.apply(lambda x: " ".join(["can" if w == "ca" else w for w in x.split()]))
    data["cues"] = (
        data.apply(get_cue, axis=1)
        # lower the cues
        .apply(lambda x: x.lower() if isinstance(x, str) else x)
    )
    # filter out sentences with multiple words
    data = data[
        (
            # handle cases where the cues are multiple words
            data.cues.apply(lambda x: not isinstance(x, list))
        )
    ]

    pos = data[data.cues.apply(lambda x: x is None)]
    pos = pos[["sent"]]
    pos["positive"] = True
    pos["subsequence"] = False

    neg = data[data.cues.apply(lambda x: x is not None)]
    neg = neg[["sent"]]
    neg["positive"] = False
    neg["subsequence"] = False

    # filter cues to a smaller subset.
    cue_filter = set(["not", "no", "never", "nor"])
    data = data[data.cues.apply(lambda c: c in cue_filter)]

    # add additional fields.
    data["scope"] = data.apply(get_scope, axis=1)
    data["full_scope"] = data.apply(get_full_scope, axis=1)

    # filter scopes that area really short.
    data = data[data.scope.apply(len) > 10]

    # generate pairs for entailment classification.
    records = data.to_dict(orient="records")
    df_sent = pd.DataFrame(list(itertools.chain.from_iterable(map(get_sentences, records))))
    df_sent["subsequence"] = True

    out = pd.concat([pos, neg, df_sent])
    out = out.sample(frac=1).reset_index(drop=True)
    out = out[out.sent.apply(filter_nli)]
    out["sent"] = out.sent.apply(fix_sent)
    pos = out[out.positive]
    neg = out[~out.positive]

    def cat(x, y):
        # shuffle
        x = x.sample(frac=1).reset_index(drop=True)
        # shuffle
        y = y.sample(frac=1).reset_index(drop=True)
        return pd.concat([x.add_suffix("_1"), y.add_suffix("_2")], axis=1)

    pos_pos = cat(pos, pos)
    neg_neg = cat(neg, neg)
    neg_pos = cat(neg, pos)
    pos_neg = cat(pos, neg)

    nli = pd.concat([pos_pos, neg_neg, neg_pos, pos_neg])
    nli["label"] = "neutral"
    nli = nli.drop_duplicates(subset=["sent_1", "sent_2"], ignore_index=True)
    # remove sentences where both sentences are the same.
    nli = nli[nli.sent_1 != nli.sent_2]
    df = nli.dropna()
    # fix outputs (de-split words like im-possible).
    # df = pd.DataFrame(map(fix_nli, records))
    # df = df.drop_duplicates(subset=["sent_1", "sent_2"], ignore_index=True)

    # match outputs for jiant
    # Filter for sentence1s that are of length 0
    # Filter if row[targ_idx] is nan
    mask = df.sent_1.str.len() > 0 & (df.sent_2.str.len() > 0) & df.label.notnull()
    df = df[mask]
    # hope: pre-tokenizing sentences will make it easier to match back to the original data
    # after it is processed by jiant. hopefully tokenizing a sentence twice won't have any
    # impact: e.g. tokenize(tokenize(x)) = tokenize(x)
    df.sent_1 = df.sent_1.apply(lambda x: tokenize_and_truncate("bert-base-cased", x, 50000))
    df.sent_2 = df.sent_2.apply(lambda x: tokenize_and_truncate("bert-base-cased", x, 50000))
    return df


def get_clauses(sent):
    """Gets the clauses from the sentence. """
    # extract constiency tree (parse)
    result = predictor.predict(sentence=sent)
    tree = Tree.fromstring(result["trees"])
    for t in tree.subtrees():
        tag = t.label()
        # Currently, we only take "S", as the other clauses
        # often can't stand on their own in practice.
        if tag == "S":  # or tag == "SINV":
            yield " ".join(t.leaves())


def get_nli(series):
    """Map instance to nli pairs. """
    negated_scope = series["full_scope"]
    scope = series["scope"]
    sent = series["sent"]
    cue = series["cues"]
    common_cue = series["common_cue"]

    negated_scope_set = set(negated_scope.split())
    scope_set = set(scope.split())

    # We look at all the clauses within the sentence, and then
    # also look at the clauses within the scope. They can overlap
    # but sometimes looking directly at the scope gives alternate
    # or more complete picture.
    clauses = itertools.chain(get_clauses(scope), get_clauses(sent))
    for c in clauses:
        # Skip the full-sentence.
        if c == "sent":
            continue

        # We approximate looking at the various sliding windows, and just make
        # sets of the words to determine the various relations between clauses
        # and scopes.
        c_set = set(c.split())

        if c_set.issubset(scope_set) and not negated_scope_set.issubset(c_set):
            # scope: he did not run
            # clause: he did run
            yield {
                "sent_1": sent,
                "sent_2": fix_sent_2(c, scope_set),
                "label": "contradiction",
                "case": "a: within scope.",
                "cue": cue,
                "common_cue": common_cue,
            }
        elif negated_scope_set.issubset(c_set):
            # input clause: he did not run, but he did dance.
            # scope: he did [not] run

            # sent_1: ...he did not run, but he did dance.
            # sent_2: he did run, but he did dance.
            # scope: he did [not] run
            sent_2 = c.replace(cue + " ", "").replace(cue, "")
            sent_2 = fix_sent_2(sent_2, scope_set)
            yield {
                "sent_1": sent,
                "sent_2": sent_2,
                "label": "contradiction",
                "case": "b: cue-removed",
                "cue": cue,
                "common_cue": common_cue,
            }
        else:
            # other cases.
            yield {
                "sent_1": sent,
                "sent_2": c,
                "label": "entailment",
                "case": "c: a S clause",
                "cue": cue,
                "common_cue": common_cue,
            }


def get_sentences(series):
    """Map instance to nli pairs. """
    negated_scope = series["full_scope"]
    scope = series["scope"]
    sent = series["sent"]
    cue = series["cues"]

    negated_scope_set = set(negated_scope.split())
    scope_set = set(scope.split())

    # We look at all the clauses within the sentence, and then
    # also look at the clauses within the scope. They can overlap
    # but sometimes looking directly at the scope gives alternate
    # or more complete picture.
    clauses = itertools.chain(get_clauses(scope), get_clauses(sent))
    for c in clauses:
        # Skip the full-sentence.
        if c == "sent":
            continue

        # We approximate looking at the various sliding windows, and just make
        # sets of the words to determine the various relations between clauses
        # and scopes.
        c_set = set(c.split())

        if c_set.issubset(scope_set) and not negated_scope_set.issubset(c_set):
            # scope: he did not run
            # clause: he did run
            yield {"sent": fix_sent_2(c, scope_set), "positive": True}

        elif negated_scope_set.issubset(c_set):
            # input clause: he did not run, but he did dance.
            # scope: he did [not] run

            # sent_1: ...he did not run, but he did dance.
            # sent_2: he did run, but he did dance.
            # scope: he did [not] run
            sent_2 = c.replace(cue + " ", "").replace(cue, "")
            sent_2 = fix_sent_2(sent_2, scope_set)
            yield {"sent": sent_2, "positive": True}
        else:
            # other cases.
            # We are asssuming the input sentence is Negative to begin with.
            yield {"sent": c, "positive": False}


def fix_sent_2(sent_2, scope_set):
    """Replace or delete NPIs.

    TODO: Re-implement scope to use the locations of the words in the original sentence.
    Currently we are implicitly making assumptions about the number of times a word appears.
    Consider `spacy:span` or something similar.
    """
    fix = {
        "ever": False,
        "either": False,
        "any": "some",
        "anyone": "someone",
        "anybody": "somebody",
    }
    fix_keys = set(fix.keys())

    def fix_word(word):
        if word not in scope_set:
            return True, word

        if word in fix_keys:
            if fix[word]:
                return True, fix[word]
            else:
                return False, None
        return True, word

    sent_2 = " ".join(word for keep, word in map(fix_word, sent_2.split()) if keep)
    return sent_2


def filter_nli(sent_2):
    """Filter bad sent_2. 
    
    Returns True if we should keep it, and False otherwise.
    """
    result = predictor.predict(sentence=sent_2)
    tree = Tree.fromstring(result["trees"])

    # bad cases
    is_not_S = tree.label() != "S"
    is_bad_S1 = len(tree) == 1 and tree[0].label() == "VP"
    is_bad_S2 = len(tree) == 2 and tree[0].label() == "PP" and tree[0].label() == "VP"

    if is_not_S or is_bad_S1 or is_bad_S2:
        return False

    return True


def fix_sent(sent):
    return sent.replace("- ", "").replace(" -", "").replace("``", '"').replace("''", '"')


def fix_nli(series):
    """Fix up the entailment instances.

    * De-split words like "im-possible".
    """
    sent_2 = series["sent_2"]
    sent_1 = series["sent_1"]
    label = series["label"]
    case = series["case"]
    cue = series["cue"]
    common_cue = series["common_cue"]
    fix = lambda x: x.replace("- ", "").replace(" -", "").replace("``", '"').replace("''", '"')
    return {
        "sent_2": fix(sent_2),
        "sent_1": fix(sent_1),
        "label": label,
        "case": case,
        "cue": cue,
        "common_cue": common_cue,
    }


def get_cue(series):
    """Extract cues using the cue indices from the sentence.
    """
    # The data is a string that should be a list of integers
    cues = ast.literal_eval(series["cues_idx"])
    # Extract all cue words.
    cue = [s for c, s in zip(cues, series["sent"].split(" ")) if c]

    if len(cue) == 1:
        return cue[0]
    elif len(cue) == 0:
        return None
    else:
        return cue


def get_scope(series):
    """Get a string of all the words in scope. """
    idx = ast.literal_eval(series["scopes_idx"])
    scope = " ".join([s for i, s in zip(idx, series["sent"].split(" ")) if i])
    return scope


def get_full_scope(series):
    """Get a string of all the words in scope including the cue, in order. """
    idx_1 = ast.literal_eval(series["scopes_idx"])
    idx_2 = ast.literal_eval(series["cues_idx"])
    scope = " ".join([s for i1, i2, s in zip(idx_1, idx_2, series["sent"].split(" ")) if i1 or i2])
    return scope


if __name__ == "__main__":
    nli()
    # random()
