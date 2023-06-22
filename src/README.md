## Instruction
You can run the script of negation scope resolution as follows:

    python main.py --split DATA --parser PARSER --rules RULES [--output_file OUTPUT_FILE_PATH]

where `DATA` is taken from {training, dev, test}, `PARSER` from {reranking, benepar, ajpar} and `RULES` from {original, modified}.
- `reranking`, `benepar` and `ajpar` represent Reranking Parser ([Charniak and Johnson, 2005](https://dl.acm.org/doi/10.3115/1219840.1219862)), Berkeley Neural Parser ([Kitaev and Klein, 2018](https://aclanthology.org/P18-1249/); [Kitaev et al., 2019](https://aclanthology.org/P19-1340/)) and Attach Juxtapose Parser ([Yang and Deng, 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/f7177163c833dff4b38fc8d2872f1ec6-Paper.pdf)), respectively.
- Regarding `RULES`, `original` applies the rules of [Read et al. (2012)](https://aclanthology.org/S12-1041/) and `modified` applies the modified rules introduced in our paper.
- If you want to specify the location and name of the output file, specify the file path using `--output_file` option. The file extension of the output file need to be `.txt`.
