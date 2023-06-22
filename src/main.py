import io, argparse
import pandas as pd
from pandas import DataFrame
from argparse import ArgumentParser, Namespace
from tree_classes import ParseNode
from io import TextIOWrapper
from typing import Optional
from format_funcs import create_string_for_nltk_tree, neg_num


def main() -> None:
    # command line arguments
    parser: ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rules", required=True, help="'original' or 'modified'")

    # constituency parser
    parser.add_argument(
        "-p", "--parser", required=True, help="'reranking' or 'benepar' or 'ajpar'"
    )

    # training, development or test
    parser.add_argument(
        "-s", "--split", required=True, help="'training' or 'dev' or 'test'"
    )

    # outputfile path (optional)
    parser.add_argument("-out", "--output_file", help="output file path")

    args: Namespace = parser.parse_args()

    arg_rules: str = args.rules
    assert arg_rules in ["original", "modified"]
    apply_modified_rules: bool = True if arg_rules == "modified" else False

    sparser: str = args.parser
    assert sparser in ["reranking", "benepar", "ajpar"]

    split: str = args.split
    assert split in ["training", "dev", "test"]

    output_file_from_args: Optional[str] = args.output_file

    input_file_path: str
    output_file_path: str

    split_name: str = "test-merged-GOLD-CUE" if split == "test" else split

    if sparser == "reranking":
        input_file_path = f"../data/cd-sco-starsem2012-st-original/cd-sco-starsem2012-st-original-{split_name}.txt"
        split_name = split_name.rstrip("-GOLD-CUE")
        output_file_path = (
            f"../output/reranking/output-reranking-{arg_rules}-rules-{split_name}.txt"
        )

    elif sparser == "benepar":
        input_file_path = f"../data/cd-sco-benepar/cd-sco-benepar-{split_name}.txt"
        split_name = split_name.rstrip("-GOLD-CUE")
        output_file_path = (
            f"../output/benepar/output-benepar-{arg_rules}-rules-{split_name}.txt"
        )

    else:
        assert sparser == "ajpar"
        input_file_path = f"../data/cd-sco-ajpar/cd-sco-ajpar-{split_name}.txt"
        split_name = split_name.rstrip("-GOLD-CUE")
        output_file_path = (
            f"../output/ajpar/output-ajpar-{arg_rules}-rules-{split_name}.txt"
        )

    # configure output file path
    if args.output_file:
        output_file_path = output_file_from_args

    data: str = ""
    sentence_counter: int = 0
    neg_sntence_counter: int = 0

    fp: TextIOWrapper = open(output_file_path, "w")
    fp.close()

    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line != "\n":
                data += line

            else:
                if data == "":
                    break

                df: DataFrame = pd.read_table(io.StringIO(data), header=None)
                tree_string: str = create_string_for_nltk_tree(df)
                neg_num_in_snt: int = neg_num(df)
                surface_list: list[str] = list(df[3])
                lemma_list: list[str] = list(df[4])
                pos_list: list[str] = list(df[5])
                syntax_list: list[str] = list(df[6])
                chapter_name: str = df.iloc[0, 0]
                sentence_num: int = int(df.iloc[0, 1])

                cue_info_list: Optional[list[list[str]]]

                if neg_num_in_snt == 0:
                    cue_info_list = None

                else:
                    cue_info_list: list[list[str]] = []
                    for i in range(neg_num_in_snt):
                        cue_info_list.append(list(df[7 + 3 * i]))

                # for sentences that do not contain negations
                if cue_info_list is None:
                    df.to_csv(
                        output_file_path,
                        sep="\t",
                        mode="a",
                        encoding="utf-8",
                        header=None,
                        index=None,
                    )
                    fp = open(output_file_path, "a", encoding="utf-8")
                    fp.write("\n")
                    fp.close()

                # for sentences that contain negation(s)
                else:
                    tree: ParseNode = ParseNode.fromstring(tree_string)
                    tree.set_leaf_constituents(
                        lemma_list=lemma_list,
                        pos_list=pos_list,
                        cue_info_list=cue_info_list,
                    )
                    tree.set_sentence_info(
                        chapter_name=chapter_name, sentence_num=sentence_num
                    )

                    scope_predictions: list[list[str]] = tree.predict_scopes(
                        apply_modified_rules=apply_modified_rules,
                    )

                    scope_predictions: list[list[str]] = tree.predict_scopes(
                        apply_modified_rules=apply_modified_rules,
                    )

                    data_list: list[list[str | int]] = [
                        [tree.chapter_name for _ in range(len(tree.leaf_constituents))],
                        [tree.sentence_num for _ in range(len(tree.leaf_constituents))],
                        [
                            leaf_constituent.id
                            for leaf_constituent in tree.leaf_constituents
                        ],
                        surface_list,
                        lemma_list,
                        pos_list,
                        syntax_list,
                    ]
                    neg_prediction_list: list[list[str]] = []
                    for i in range(len(scope_predictions)):
                        neg_prediction_list.append(list(df[7 + 3 * i]))
                        neg_prediction_list.append(scope_predictions[i])
                        neg_prediction_list.append(
                            ["#" for _ in range(len(scope_predictions[i]))]
                        )

                    list_for_output: list[list[str | int]] = (
                        data_list + neg_prediction_list
                    )
                    df_for_output: DataFrame = pd.DataFrame(list_for_output).T

                    df_for_output.to_csv(
                        output_file_path,
                        sep="\t",
                        mode="a",
                        encoding="utf-8",
                        header=None,
                        index=None,
                    )
                    fp = open(output_file_path, "a", encoding="utf-8")
                    fp.write("\n")
                    fp.close()

                    neg_sntence_counter += 1

                data = ""
                sentence_counter += 1


if __name__ == "__main__":
    main()
