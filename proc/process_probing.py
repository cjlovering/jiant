import os

import numpy as np
import plac


@plac.opt("prop", "property name", choices=["gap", "isl"])
@plac.opt("split", "co occurence rate", choices=["strong_probing", "weak_probing"])
@plac.opt("folder", "folder name")
def main(prop="gap", split="strong_probing", folder="2020-08-02"):
    f_name = f"{folder}/{prop}/{split}/log.log"

    with open(f_name, "r") as f:
        valid_list = []
        for l in f.readlines():
            l_split = l.split(" ")
            if (
                len(l_split) > 3
                and (
                    l_split[3].startswith(f"{prop}_probing")
                    or l_split[3].startswith(f"{prop}_finetune")
                )
                and l_split[3].endswith("accuracy:")
            ):
                valid_list.append(float(l_split[len(l_split) - 1]))

    error = 1 - np.array(valid_list)
    print("prop\tsplit\tepochs\tauc\tbest")
    print(f"{prop}\t{split}\t{len(error)}\t{np.sum(error)}\t{np.min(error)}")


if __name__ == "__main__":
    plac.call(main)