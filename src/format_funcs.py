from pandas import DataFrame


def insert_spaces_into_syntax(syntax_str: str) -> str:
    """
    ===== Example =====
    arg syntax_str: (S(NP(NP***)*(SBAR(WHNP*)(S(VP*(ADVP*)(ADJP**(PP*(NP**)))*(ADVP*(PP*(NP(NP*(ADJP**)*)(SBAR(WHADVP*)(S(NP*)(VP*(ADVP*)(NP**))))))))))*)(VP*(VP*(PP*(NP***))))*)
    ret : (S (NP (NP * * *) * (SBAR (WHNP *) (S (VP * (ADVP *) (ADJP * * (PP * (NP * *))) * (ADVP * (PP * (NP (NP * (ADJP * *) *) (SBAR (WHADVP *) (S (NP *) (VP * (ADVP *) (NP * *)))))))))) *) (VP * (VP * (PP * (NP * * *)))) *)
    """
    syntax_str_space: str = ""

    for i, t in enumerate(syntax_str):
        if t == "(":
            syntax_str_space += t
        elif i < len(syntax_str) - 1 and (
            syntax_str[i + 1] == ")" or syntax_str[i + 1].isalpha()
        ):
            syntax_str_space += t
        else:
            syntax_str_space += t + " "

    return syntax_str_space


def insert_pos_and_surface_into_syntax(
    syntax_str_space: str, df: DataFrame, pos_col: int = 5, surface_col: int = 3
) -> str:
    """
    ===== Example =====
    arg syntax_str_space: (S (NP (NP * * *) * (SBAR (WHNP *) (S (VP * (ADVP *) (ADJP * * (PP * (NP * *))) * (ADVP * (PP * (NP (NP * (ADJP * *) *) (SBAR (WHADVP *) (S (NP *) (VP * (ADVP *) (NP * *)))))))))) *) (VP * (VP * (PP * (NP * * *)))) *)
    ret: (S (NP (NP ( NNP Mr.) ( NNP Sherlock) ( NNP Holmes)) ( , ,) (SBAR (WHNP ( WP who)) (S (VP ( VBD was) (ADVP ( RB usually)) (ADJP ( RB very) ( JJ late) (PP ( IN in) (NP ( DT the) ( NNS mornings)))) ( , ,) (ADVP ( RB save) (PP ( IN upon) (NP (NP ( DT those) (ADJP ( RB not) ( JJ infrequent)) ( NNS occasions)) (SBAR (WHADVP ( WRB when)) (S (NP ( PRP he)) (VP ( VBD was) (ADVP ( RB up)) (NP ( DT all) ( NN night))))))))))) ( , ,)) (VP ( VBD was) (VP ( VBN seated) (PP ( IN at) (NP ( DT the) ( NN breakfast) ( NN table))))) ( . .))
    """

    tree_string: str = ""
    token_num: int = 0

    surface_modified: str
    for t in syntax_str_space:
        if t == "*":
            if df.iloc[token_num, surface_col] == "(":
                surface_modified = "-LRB-"
            elif df.iloc[token_num, surface_col] == ")":
                surface_modified = "-RRB-"
            else:
                surface_modified = df.iloc[token_num, surface_col]

            tree_string += (
                "( " + df.iloc[token_num, pos_col] + " " + surface_modified + ")"
            )
            token_num += 1
        else:
            tree_string += t

    return tree_string


def create_string_for_nltk_tree(df: DataFrame) -> str:
    """
    ===== Example =====
    ret: (S (NP (NP ( NNP Mr.) ( NNP Sherlock) ( NNP Holmes)) ( , ,) (SBAR (WHNP ( WP who)) (S (VP ( VBD was) (ADVP ( RB usually)) (ADJP ( RB very) ( JJ late) (PP ( IN in) (NP ( DT the) ( NNS mornings)))) ( , ,) (ADVP ( RB save) (PP ( IN upon) (NP (NP ( DT those) (ADJP ( RB not) ( JJ infrequent)) ( NNS occasions)) (SBAR (WHADVP ( WRB when)) (S (NP ( PRP he)) (VP ( VBD was) (ADVP ( RB up)) (NP ( DT all) ( NN night))))))))))) ( , ,)) (VP ( VBD was) (VP ( VBN seated) (PP ( IN at) (NP ( DT the) ( NN breakfast) ( NN table))))) ( . .))
    """
    syntax: list[str] = list(df[6])
    syntax_str: str = "".join(syntax)

    syntax_str_space: str = insert_spaces_into_syntax(syntax_str)
    tree_string: str = insert_pos_and_surface_into_syntax(syntax_str_space, df)

    return tree_string


def neg_num(df: DataFrame) -> int:
    col_num: int = len(df.columns)
    if col_num == 8:
        return 0

    neg_num = int((col_num - 7) / 3)

    return neg_num
