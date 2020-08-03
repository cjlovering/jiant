import jsonlines
import pandas as pd
import plac
import sklearn.metrics


@plac.opt("prop", "property name", choices=["gap", "isl"])
@plac.opt("folder", "folder name")
@plac.opt("rate", "co occurence rate", choices=["0", "1", "5"])
def main(prop="gap", rate="0", folder="2020-08-02"):
    # Load initial data with additional meta information.
    with jsonlines.open(f"../data_gen/{prop}/{prop}_test.jsonl") as reader:
        data = [obj for obj in reader]

    # Prepare results.
    df = pd.read_csv(f"./{folder}/{prop}/{rate}/{prop}_finetune_{rate}_test.tsv", sep="\t")
    df["sentence"] = df["sentence_1"].apply(untokenize)
    new_cols = df.apply(lambda x: extract_meta(x, data), axis=1, result_type="expand")
    df = pd.concat([df, new_cols], axis="columns")
    df["error"] = df["label"] != df["true_label"]
    df["bad"] = ((df.section == "both") | (df.section == "bad-only")).astype(int)
    both = df[df.section == "both"]
    neither = df[df.section == "neither"]
    # good = df[df.section == "good-only"]
    bad = df[df.section == "bad-only"]

    # Print results.
    score = lambda x: x["error"].mean()
    I_pred_true = sklearn.metrics.mutual_info_score(df["true_label"], df["prediction"])
    I_pred_dist = sklearn.metrics.mutual_info_score(df["bad"], df["prediction"])
    print("prop\trate\ttest\tboth\tneither\tbad\tI(pred;true)\tI(pred;bad)")
    print(
        f"{prop}\t{rate}\t{score(df)}\t{score(both)}\t{score(neither)}\t{score(bad)}\t{I_pred_true}\t{I_pred_dist}"
    )


def untokenize(sentence):
    """untokenizes a sentence to its orignal form.
    """
    sentence = sentence.replace(" ##", "")
    sentence = sentence.replace(" ' ", "'")
    sentence = sentence.replace(" &apos;", "'")
    return sentence


def extract_meta(row, data):
    """Matches the sentence from `row` with the corresponding dict in `data`.
    
    The result is used to add new columns to the dataframe.
    """
    sentence = row["sentence"]
    for sent in data:
        # TODO: some of the sentences look incomplete for some reason (22/750)
        if sentence in sent["sentence"]:
            return sent


if __name__ == "__main__":
    plac.call(main)
