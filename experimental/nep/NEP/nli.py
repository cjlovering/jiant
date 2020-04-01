import ast
import itertools
import pandas as pd
from collections import Counter
from nltk import Tree

# We use the allennlp parser (I liked the outputs I sampled.)
# https://demo.allennlp.org/constituency-parsing/MTU5NjQxOQ==

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
    for f in files:
        df = pipeline(f)
        df.to_csv(
            f"./random/{f}.tsv",
            index=False,
            sep="\t",
            columns=["premise", "hypothesis", "label", "case"],
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
    for f in files:
        df = pipeline_random(f)
        df.to_csv(f"./random/{f}.tsv", index=False, sep="\t")


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
    df = data[data.cues.apply(lambda c: c in cue_filter)]

    # filter scopes that area really short.
    df = df[df.scope.apply(len) > 10]

    # generate pairs for entailment classification.
    records = df.to_dict(orient="records")
    df_nli = pd.DataFrame(list(itertools.chain.from_iterable(map(get_nli, records))))

    # filter hypotheses s.t. they are (more) well-formed.
    df_nli = df_nli[df_nli.hypothesis.apply(filter_nli)]

    # fix outputs (de-split words like im-possible).
    records = df_nli.to_dict(orient="records")
    df = pd.DataFrame(map(fix_nli, records))
    df = df.drop_duplicates(subset=["premise", "hypothesis"], ignore_index=True)
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
    nli = nli.dropna()
    # fix outputs (de-split words like im-possible).
    # df = pd.DataFrame(map(fix_nli, records))
    # df = df.drop_duplicates(subset=["sent_1", "sent_2"], ignore_index=True)
    return nli


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
                "premise": sent,
                "hypothesis": fix_hypothesis(c, scope_set),
                "label": "contradiction",
                "case": "a: within scope.",
            }
        elif negated_scope_set.issubset(c_set):
            # input clause: he did not run, but he did dance.
            # scope: he did [not] run

            # premise: ...he did not run, but he did dance.
            # hypothesis: he did run, but he did dance.
            # scope: he did [not] run
            hypothesis = c.replace(cue + " ", "").replace(cue, "")
            hypothesis = fix_hypothesis(hypothesis, scope_set)
            yield {
                "premise": sent,
                "hypothesis": hypothesis,
                "label": "contradiction",
                "case": "b: cue-removed",
            }
        else:
            # other cases.
            yield {"premise": sent, "hypothesis": c, "label": "entailment", "case": "c: a S clause"}


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
            yield {"sent": fix_hypothesis(c, scope_set), "positive": True}

        elif negated_scope_set.issubset(c_set):
            # input clause: he did not run, but he did dance.
            # scope: he did [not] run

            # premise: ...he did not run, but he did dance.
            # hypothesis: he did run, but he did dance.
            # scope: he did [not] run
            hypothesis = c.replace(cue + " ", "").replace(cue, "")
            hypothesis = fix_hypothesis(hypothesis, scope_set)
            yield {"sent": hypothesis, "positive": True}
        else:
            # other cases.
            # We are asssuming the input sentence is Negative to begin with.
            yield {"sent": c, "positive": False}


def fix_hypothesis(hypothesis, scope_set):
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

    hypothesis = " ".join(word for keep, word in map(fix_word, hypothesis.split()) if keep)
    return hypothesis


def filter_nli(hypothesis):
    """Filter bad hypothesis. 
    
    Returns True if we should keep it, and False otherwise.
    """
    result = predictor.predict(sentence=hypothesis)
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
    hypothesis = series["hypothesis"]
    premise = series["premise"]
    label = series["label"]
    case = series["case"]
    fix = lambda x: x.replace("- ", "").replace(" -", "").replace("``", '"').replace("''", '"')
    return {"hypothesis": fix(hypothesis), "premise": fix(premise), "label": label, "case": case}


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
    # nli()
    random()
