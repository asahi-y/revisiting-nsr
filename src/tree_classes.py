import re
from nltk import ParentedTree
from typing import Union
from typing import Optional

WH_PHRASE_TAGS: list[str] = ["WHADJP", "WHADVP", "WHNP", "WHPP"]
VB_ASTERISK_POS_LIST: list[str] = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
S_ASER_LIST: list[str] = ["S", "SBAR", "SBARQ", "SINV", "SQ"]
PUNCTUATION_LIST: list[str] = [
    ",",
    ".",
    "''",
    "``",
    ":",
    "-LRB-",
    "-RRB-",
    "HYPH",
    "NFP",
]


class LeafConstitunent:
    def __init__(
        self,
        tree: "ParseNode",
        surface: str,
        id: Optional[int] = None,
        lemma: Optional[str] = None,
        pos: Optional[str] = None,
        cue_num: Optional[int] = None,
        cue_info_list: Optional[list[str]] = None,
    ) -> None:
        self.id: Optional[int] = id
        self.tree: "ParseNode" = tree
        self.surface: str = surface
        self.lemma: Optional[str] = lemma
        self.pos: Optional[str] = pos
        self.is_scope_pred: bool = True
        self.cue_num: Optional[int] = cue_num
        self.cue_info_list: Optional[list[str]] = cue_info_list

    def __str__(self) -> str:
        return str(
            {
                "id": self.id,
                "surface": self.surface,
                "lemma": self.lemma,
                "POS": self.pos,
                "is_scope_pred": self.is_scope_pred,
                "cue_info_list": self.cue_info_list,
                "tree (label)": self.tree.label(),
            }
        )

    def __repr__(self) -> str:
        return str(
            {
                "id": self.id,
                "surface": self.surface,
                "lemma": self.lemma,
                "POS": self.pos,
                "is_scope_pred": self.is_scope_pred,
                "cue_info_list": self.cue_info_list,
                "tree (label)": self.tree.label(),
            }
        )


class ParseNode(ParentedTree):
    def __init__(self, node, children: Optional[ParentedTree] = None) -> None:
        super().__init__(node, children)

        self.leaf_constituents: Optional[list[LeafConstitunent]] = None

        self.is_leaf: bool = False
        self.leaf_info: Optional[LeafConstitunent] = None
        self.chapter_name: Optional[str] = None
        self.sentence_num: Optional[int] = None
        self.applied_rule_id: Optional[int] = None
        self.predicted_tree_info_list: Optional[list[dict]] = None

    # Override (add type annotation)
    def parent(self) -> Union["ParseNode", None]:
        return self._parent

    def _create_leaf_constituents(self) -> list[LeafConstitunent]:
        self.leaf_constituents: list[LeafConstitunent] = []
        for child in self:
            assert isinstance(child, ParseNode), "Input tree is invalid"

            if child.height() > 2:
                self.leaf_constituents.extend(child._create_leaf_constituents())

            else:
                self.leaf_constituents.append(
                    LeafConstitunent(tree=child, surface=child[0])
                )

        return self.leaf_constituents

    def _set_id_to_leaf_constituents(self) -> None:
        for i, leaf in enumerate(self.leaf_constituents):
            leaf: LeafConstitunent
            leaf.id = i

    def _set_lemma_to_leaf_constituents(self, lemma_list: list[str]) -> None:
        assert len(self.leaf_constituents) == len(
            lemma_list
        ), "Length of lemma_list is invalid."

        for i, leaf in enumerate(self.leaf_constituents):
            leaf: LeafConstitunent
            leaf.lemma = lemma_list[i]

    def _set_pos_to_leaf_constituents(self, pos_list: list[str]) -> None:
        assert len(self.leaf_constituents) == len(
            pos_list
        ), "Length of pos_list is invalid."

        for i, leaf in enumerate(self.leaf_constituents):
            leaf: LeafConstitunent
            leaf.pos = pos_list[i]

    def _set_cue_info_to_leaf_constituents(
        self, cue_info_list: list[list[str]]
    ) -> None:
        """
        Set cue_info_list to each element of self.leaf_constituents.
        """
        assert len(self.leaf_constituents) == len(
            cue_info_list[0]
        ), "Length of cue_info_list[0] is invalid."

        for i, leaf in enumerate(self.leaf_constituents):
            leaf: LeafConstitunent
            leaf.cue_info_list = [
                cue_info_list[j][i] for j in range(len(cue_info_list))
            ]

    def _set_leaf_info(self) -> None:
        """
        Set self.is_leaf and self.leaf_info to ParseNode object whose height is 2.
        """
        assert (
            self.leaf_constituents is not None
        ), "self.leaf_constituents is not defined."

        subtrees_leaf: list[ParseNode] = self.subtrees(lambda t: t.height() == 2)

        for i, leaf in enumerate(subtrees_leaf):
            leaf: ParseNode
            leaf.is_leaf = True
            leaf.leaf_info = self.leaf_constituents[i]

        assert i + 1 == len(
            self.leaf_constituents
        ), "Length of subtree_leaf and that of self.leaf_constituents must be the same."

    def set_leaf_constituents(
        self,
        lemma_list: Optional[list[str]] = None,
        pos_list: Optional[list[str]] = None,
        cue_info_list: Optional[list[list[str]]] = None,
    ) -> None:
        self._create_leaf_constituents()
        self._set_id_to_leaf_constituents()

        if lemma_list:
            self._set_lemma_to_leaf_constituents(lemma_list)

        if pos_list:
            self._set_pos_to_leaf_constituents(pos_list)

        if cue_info_list:
            self._set_cue_info_to_leaf_constituents(cue_info_list)

        self._set_leaf_info()

    def set_sentence_info(
        self, chapter_name: Optional[str] = None, sentence_num: Optional[int] = None
    ) -> None:
        """
        Set chapter_name and sentence_num to a ParseNode object.
        """
        if chapter_name is not None:
            self.chapter_name = chapter_name

        if sentence_num is not None:
            self.sentence_num = sentence_num

    # =============================================================================== #
    # =============================== Search Methods ================================ #
    # =============================================================================== #
    def search_ancestors(
        self,
        target_label_list: list[str],
        terminal: Optional["ParseNode"] = None,
        excluded_subtree: Optional["ParseNode"] = None,
    ) -> Optional["ParseNode"]:
        if self.parent() == None:
            return None
        # Return None if reaching the terminal node.
        elif terminal and self.parent().height() >= terminal.height():
            return None

        if excluded_subtree:
            for tree in self.parent().subtrees():
                if tree.is_same_tree(excluded_subtree):
                    return None

        if self.parent().label() in target_label_list:
            return self.parent()
        else:
            return self.parent().search_ancestors(
                target_label_list, terminal, excluded_subtree
            )

    def search_parent(self, target_label_list: list[str]) -> Optional["ParseNode"]:
        if self.parent() == None:
            return None
        elif self.parent().label() in target_label_list:
            return self.parent()
        else:
            return None

    def search_children(
        self, target_label_list: list[str]
    ) -> Optional[list["ParseNode"]]:
        ret_children: list["ParseNode"] = []

        if not isinstance(self, ParseNode):
            return None

        for child in self:
            if child.label() in target_label_list:
                ret_children.append(child)

        if len(ret_children) == 0:
            return None

        return ret_children

    def search_siblings(
        self, target_label_list: list[str]
    ) -> Union[list["ParseNode"], None]:
        ret_siblings: list["ParseNode"] = []

        if not isinstance(self, ParseNode):
            return None

        right_sibling: Optional[ParseNode] = self.right_sibling()

        # Search right siblings.
        while right_sibling is not None:
            right_sibling: ParseNode
            if right_sibling.label() in target_label_list:
                ret_siblings.append(right_sibling)

            right_sibling = right_sibling.right_sibling()

        left_sibling: Union[ParseNode, None] = self.left_sibling()

        # Search left siblings.
        while left_sibling is not None:
            left_sibling: ParseNode
            if left_sibling.label() in target_label_list:
                ret_siblings.append(left_sibling)

            left_sibling = left_sibling.left_sibling()

        if len(ret_siblings) == 0:
            return None

        return ret_siblings

    # =============================================================================== #
    # ========================= Tool Methods ======================================== #
    # =============================================================================== #

    def negation_num_in_snt(self) -> int:
        neg_num: int = (
            0
            if self.leaf_constituents[0].cue_info_list is None
            else len(self.leaf_constituents[0].cue_info_list)
        )
        return neg_num

    def in_scope_prediction_head(self) -> LeafConstitunent:
        """
        Method to get the first LeafConstituent object whose in_scope_pred is True.
        """
        for leafconstituent in self.leaf_constituents:
            if leafconstituent.is_scope_pred:
                return leafconstituent
        else:
            raise ValueError(
                f"{self.root().chapter_name} {self.root().sentence_num}: all constituents are out of scope prediction!"
            )

    def in_scope_prediction_tail(self) -> LeafConstitunent:
        """
        Method to get the last LeafConstituent object whose in_scope_pred is True.
        """
        for leafconstituent in self.leaf_constituents[::-1]:
            if leafconstituent.is_scope_pred:
                return leafconstituent
        else:
            raise ValueError(
                f"{self.root().chapter_name} {self.root().sentence_num}: all constituents are out of scope prediction!"
            )

    def remove_tokens_from_scope_prediction(
        self,
        head_token_id: int,
        tail_token_id: int,
        neg_num_in_snt: Optional[int] = None,
        not_single_cue_token=False,
    ) -> None:
        """
        Method to remove certain parts of the ParseNode object from scope prediction.
        args:
            head_token_id: id of the first token in the removal range
            tail_token_id: id of the last token in the removal range
            neg_num_in_snt: negation id in the sentence
            not_single_cue_token: True if containing multi-word cue
        """
        for leaf_constituent in self.leaf_constituents:
            if head_token_id <= leaf_constituent.id <= tail_token_id:
                leaf_constituent.is_scope_pred = False

    def _reset_scope_pred(self) -> None:
        for leaf_constituent in self.leaf_constituents:
            leaf_constituent.is_scope_pred = True

        self.applied_rule_id = None

    def right_siblings(self) -> Optional[list["ParseNode"]]:
        tree: Optional[ParseNode] = self
        right_siblings: list[ParseNode] = []

        while tree.right_sibling():
            right_siblings.append(tree.right_sibling())
            tree = tree.right_sibling()

        if len(right_siblings) == 0:
            return None
        else:
            return right_siblings

    def left_siblings(self) -> Optional[list["ParseNode"]]:
        tree: Optional[ParseNode] = self
        left_siblings: list[ParseNode] = []

        while tree.left_sibling():
            left_siblings.append(tree.left_sibling())
            tree = tree.left_sibling()

        if len(left_siblings) == 0:
            return None
        else:
            return left_siblings

    def negation_cue_trees(self, neg_num_in_snt: int) -> list["ParseNode"]:
        """
        Method to get the ParseNode object that directly dominates the cue.
        """
        negation_cue_trees: list[ParseNode] = []
        for leaf_constituent in self.leaf_constituents:
            if leaf_constituent.cue_info_list[neg_num_in_snt] != "_":
                negation_cue_trees.append(leaf_constituent.tree)

        assert len(negation_cue_trees) > 0, "Not existing corresponding negation."

        return negation_cue_trees

    def negation_cue_token_ids(self, neg_num_in_snt: int) -> list[int]:
        negation_cue_token_ids: list[int] = []
        for leaf_constituent in self.leaf_constituents:
            if leaf_constituent.cue_info_list[neg_num_in_snt] != "_":
                negation_cue_token_ids.append(leaf_constituent.id)

        assert len(negation_cue_token_ids) > 0, "Not existing corresponding negation."

        return negation_cue_token_ids

    def is_same_tree(self, tree: "ParseNode") -> bool:
        """
        Method to determine if a given tree is the same as self, including leaves.
        """
        token_id_list_self: list[int]
        if self.height() > 2:
            token_id_list_self = [leaf.id for leaf in self.leaf_constituents]
        else:
            token_id_list_self = [self.leaf_info.id]

        token_id_list_tree: list[int]
        if tree.height() > 2:
            token_id_list_tree = [leaf.id for leaf in tree.leaf_constituents]
        else:
            token_id_list_tree = [tree.leaf_info.id]

        if token_id_list_self == token_id_list_tree:
            return True
        else:
            return False

    # =============================================================================== #
    # ========================= Scope Resolution Heuristics Methods ================= #
    # =============================================================================== #

    def _cue_token_id_list(self) -> Optional[list[list[int]]]:
        """
        e.g.,
        Consider the sentence that contains three negation cues;
            - not (id: 3)
            - im (id: 8)
            - no more (id: 12, 13)
        then, the returned value is as follows:
        [[3], [8], [12, 13]]
        """
        if self.negation_num_in_snt == 0:
            return None

        cue_token_id_list: list[list[int]] = [
            [] for _ in range(self.negation_num_in_snt())
        ]
        for leaf_constituent in self.leaf_constituents:
            for i, cue in enumerate(leaf_constituent.cue_info_list):
                cue: str
                if cue != "_":
                    cue_token_id_list[i].append(leaf_constituent.id)

        return cue_token_id_list

    def _apply_heuristics(
        self, cue_leaf_constituent: LeafConstitunent, apply_modified_rules: bool = False
    ) -> Optional["ParseNode"]:
        cue_pos: str = cue_leaf_constituent.pos
        scope_tree: Optional[ParseNode]

        if cue_pos == "RB":
            scope_tree = ScopeResolutionHeuristics.apply_heuristics_to_RB(
                cue_leaf_constituent, apply_modified_rules=apply_modified_rules
            )
        elif cue_pos == "DT":
            scope_tree = ScopeResolutionHeuristics.apply_heuristics_to_DT(
                cue_leaf_constituent
            )
        elif cue_pos == "JJ":
            scope_tree = ScopeResolutionHeuristics.apply_heuristics_to_JJ(
                cue_leaf_constituent
            )
        elif cue_pos == "UH":
            scope_tree = ScopeResolutionHeuristics.apply_heuristics_to_UH(
                cue_leaf_constituent
            )
        elif cue_pos == "IN":
            scope_tree = ScopeResolutionHeuristics.apply_heuristics_to_IN(
                cue_leaf_constituent
            )
        elif cue_pos == "NN":
            scope_tree = ScopeResolutionHeuristics.apply_heuristics_to_NN(
                cue_leaf_constituent
            )
        elif cue_pos == "CC":
            scope_tree = ScopeResolutionHeuristics.apply_heuristics_to_CC(
                cue_leaf_constituent
            )
        else:
            scope_tree = None

        return scope_tree

    def _create_scope_predictions(
        self,
        neg_num_in_snt: int,
        token_num_in_snt: int,
        leaf_constituents_in_sope: Optional[list[LeafConstitunent]] = None,
    ) -> list[str]:
        """
        e.g.,
        Consider the following sentence.
        "He was not happy but sad ."
        If 'He', 'was' and 'happy' were predicted the scope of the cue 'not',
        then the returned value is as follows:
        ['_', 'was', '_', 'happy', '_', '_']
        """
        scope_prediction: list[str] = ["" for _ in range(token_num_in_snt)]

        if leaf_constituents_in_sope is None:
            return ["_" for _ in range(token_num_in_snt)]

        for leaf_constituent in leaf_constituents_in_sope:
            if leaf_constituent.cue_info_list[neg_num_in_snt] != "_":
                # handling the case "less" + "ly"
                if (
                    leaf_constituent.cue_info_list[neg_num_in_snt]
                    != leaf_constituent.surface
                    and "lessly" in leaf_constituent.surface
                ):
                    token_surface: str = leaf_constituent.surface
                    scope_part: str = re.sub("lessly", "", token_surface, count=1)
                    scope_prediction[leaf_constituent.id] = scope_part
                # handling the case "less" + "ness"
                elif (
                    leaf_constituent.cue_info_list[neg_num_in_snt]
                    != leaf_constituent.surface
                    and "lessness" in leaf_constituent.surface
                ):
                    token_surface: str = leaf_constituent.surface
                    scope_part: str = re.sub("lessness", "", token_surface, count=1)
                    scope_prediction[leaf_constituent.id] = scope_part
                elif (
                    leaf_constituent.cue_info_list[neg_num_in_snt]
                    != leaf_constituent.surface
                ):
                    token_surface: str = leaf_constituent.surface
                    cue_part: str = leaf_constituent.cue_info_list[neg_num_in_snt]
                    scope_part: str = re.sub(cue_part, "", token_surface, count=1)
                    scope_prediction[leaf_constituent.id] = scope_part
                else:
                    scope_prediction[leaf_constituent.id] = "_"
            else:
                scope_prediction[leaf_constituent.id] = leaf_constituent.surface

        # give "_" to out-of-scope tokens
        for i in range(len(scope_prediction)):
            if scope_prediction[i] == "":
                scope_prediction[i] = "_"

        return scope_prediction

    def _apply_default_scope_prediction(
        self, root: "ParseNode", cue_left_id: int, cue_right_id: int
    ) -> list[LeafConstitunent]:
        start: int = cue_left_id
        while root.leaf_constituents[start].pos not in PUNCTUATION_LIST and start >= 0:
            start -= 1

        end: int = cue_right_id
        while (
            root.leaf_constituents[end].pos not in PUNCTUATION_LIST
            and end < len(root.leaf_constituents) - 1
        ):
            end += 1

        leaf_constituents: list[LeafConstitunent] = []
        for i in range(start + 1, end):
            leaf_constituents.append(root.leaf_constituents[i])

        return leaf_constituents

    # =============================================================================== #
    # ============ Slackening Rules and Punctuations Removal Methods ================ #
    # =============================================================================== #
    def apply_slackening_rules(
        self,
        neg_num_in_snt: int,
        initial_and_final_punc_removal: bool = True,
        apply_modified_rules: bool = False,
        not_single_cue_token: bool = False,
    ) -> None:
        if self.height() == 2:
            return None

        # get the ParseNode that directly dominates the cue
        negation_cue_tree: ParseNode = self.negation_cue_trees(neg_num_in_snt)[0]
        negation_cue_token_id: int = self.negation_cue_token_ids(neg_num_in_snt)[0]

        slackened_parts: ParseNode | LeafConstitunent | None = None

        first_processing_flag: bool = True

        roop_counter: int = 0
        max_iter: int = 30
        while first_processing_flag or slackened_parts:
            if roop_counter > max_iter:
                break
            if slackened_parts != None:
                if isinstance(slackened_parts, ParseNode):
                    head_token_id: int = slackened_parts.leaf_constituents[0].id
                    tail_token_id: int = slackened_parts.leaf_constituents[-1].id
                else:
                    head_token_id: int = slackened_parts.id
                    tail_token_id: int = slackened_parts.id

                self.remove_tokens_from_scope_prediction(
                    head_token_id,
                    tail_token_id,
                    neg_num_in_snt,
                    not_single_cue_token=not_single_cue_token,
                )

            # remove constituent-initial and final punctuations
            initial_or_final_punc: Optional[LeafConstitunent]
            if initial_and_final_punc_removal:
                initial_or_final_punc = self._get_initial_or_final_token(
                    PUNCTUATION_LIST, neg_num_in_snt
                )
            else:
                initial_or_final_punc = None

            # CC and following conjuncts
            cc_slackened: Optional[LeafConstitunent]
            conjunct_slackend: Optional[ParseNode | LeafConstitunent]
            cc_slackened, conjunct_slackend = self._get_cc_and_conjuncts_right(
                negation_cue_token_id
            )

            # S* right of the cue if delimited by punctuaions
            punc_s_ast_right: Optional[
                ParseNode
            ] = self._get_punc_delimited_tree_right_of_cue(
                target_labels=S_ASER_LIST,
                excluded_subtree=negation_cue_tree,
                negation_cue_token_id=negation_cue_token_id,
            )

            # initial SBAR
            initial_sbar: Optional[
                ParseNode
            ] = self.in_scope_prediction_head().tree.search_ancestors(
                ["SBAR"], terminal=self, excluded_subtree=negation_cue_tree
            )

            # initial PP (modified rule)
            initial_pp: Optional[ParseNode] = None
            if apply_modified_rules:
                initial_pp = self.in_scope_prediction_head().tree.search_ancestors(
                    ["PP"], terminal=self, excluded_subtree=negation_cue_tree
                )

            # punctuation delimited NPs
            initial_or_final_punc_np: Optional[
                ParseNode
            ] = self._get_punc_delemited_phrase(
                ["NP"], excluded_subtree=negation_cue_tree
            )

            # initial ADVP, INTJ
            initial_advp_intj: Optional[
                ParseNode
            ] = self.in_scope_prediction_head().tree.search_ancestors(
                ["ADVP", "INTJ"], terminal=self, excluded_subtree=negation_cue_tree
            )

            if (
                initial_advp_intj
                and len(initial_advp_intj.leaf_constituents) == 1
                and initial_advp_intj.leaf_constituents[0].cue_info_list[neg_num_in_snt]
                != "_"
            ):
                initial_advp_intj = None

            # initial RB, CC, UH
            initial_rb_cc_uh: Optional[LeafConstitunent] = (
                self.in_scope_prediction_head()
                if self.in_scope_prediction_head().pos in ["RB", "CC", "UH"]
                and self.in_scope_prediction_head().cue_info_list[neg_num_in_snt] == "_"
                else None
            )

            slackened_parts = (
                initial_or_final_punc
                or cc_slackened
                or conjunct_slackend
                or punc_s_ast_right
                or initial_or_final_punc
                or initial_sbar
                or initial_advp_intj
                or initial_rb_cc_uh
                or initial_or_final_punc_np
                or initial_pp
            )

            first_processing_flag = False
            roop_counter += 1

    def _get_cc_and_conjuncts_right(
        self, negation_cue_token_id: int
    ) -> tuple[
        Optional[LeafConstitunent],
        Optional[Union["ParseNode", LeafConstitunent]],
    ]:
        cc_slackened: Optional[ParseNode] = None
        conjuncts_slackend: list[ParseNode | LeafConstitunent]
        conjunct_slackend: Optional[ParseNode | LeafConstitunent] = None

        tree_pnt: Optional[ParseNode] = (
            self.root().leaf_constituents[negation_cue_token_id].tree
        )

        while tree_pnt != self:
            # not process if the parent is NP
            if tree_pnt.parent() and tree_pnt.parent().label() == "NP":
                tree_pnt = tree_pnt.parent()
                continue
            else:
                if tree_pnt.right_siblings() is None:
                    tree_pnt = tree_pnt.parent()
                    continue

                # search right siblings
                for right_sib in tree_pnt.right_siblings():
                    if right_sib.label() == "CC":
                        cc_slackened = right_sib.leaf_info
                        break

                if cc_slackened:
                    cc_slackened: LeafConstitunent
                    conjuncts_slackend = cc_slackened.tree.right_siblings()

                    if not cc_slackened.is_scope_pred:
                        cc_slackened = None

                    if conjuncts_slackend:
                        for conjunct in conjuncts_slackend:
                            scope_pred_list: list[bool] = (
                                [
                                    leaf_constituent.is_scope_pred
                                    for leaf_constituent in conjunct.leaf_constituents
                                ]
                                if conjunct.height() > 2
                                else [conjunct.leaf_info.is_scope_pred]
                            )
                            if True in scope_pred_list:
                                conjunct_slackend = (
                                    conjunct
                                    if not conjunct.is_leaf
                                    else conjunct.leaf_info
                                )
                                break

                if cc_slackened or conjunct_slackend:
                    break
                else:
                    tree_pnt = tree_pnt.parent()

        return cc_slackened, conjunct_slackend

    def _get_punc_delemited_phrase(
        self,
        target_phrases: list[str],
        excluded_subtree: Optional["ParseNode"] = None,
    ) -> Optional["ParseNode"]:
        initial_np: Optional[
            ParseNode
        ] = self.in_scope_prediction_head().tree.search_ancestors(
            target_phrases, terminal=self, excluded_subtree=excluded_subtree
        )
        if initial_np:
            np_tail_id: int = initial_np.leaf_constituents[-1].id
            initial_punc_np: Optional[ParseNode] = (
                initial_np
                if np_tail_id < len(self.root().leaf_constituents) - 1
                and self.root().leaf_constituents[np_tail_id + 1].pos
                in PUNCTUATION_LIST
                else None
            )
        else:
            initial_punc_np = None

        final_np: Optional[
            ParseNode
        ] = self.in_scope_prediction_tail().tree.search_ancestors(
            target_phrases, terminal=self, excluded_subtree=excluded_subtree
        )
        if final_np:
            np_head_id: int = final_np.leaf_constituents[0].id
            final_punc_np: Optional[ParseNode] = (
                final_np
                if np_head_id > 0
                and self.root().leaf_constituents[np_head_id - 1].pos
                in PUNCTUATION_LIST
                else None
            )
        else:
            final_punc_np = None

        initial_or_final_punc_np: Optional[ParseNode] = initial_punc_np or final_punc_np

        return initial_or_final_punc_np

    def _get_initial_or_final_token(
        self,
        target_pos_list: list[str],
        neg_num_in_snt: int,
    ) -> Optional[LeafConstitunent]:
        initial_punc: Optional[LeafConstitunent] = (
            self.in_scope_prediction_head()
            if self.in_scope_prediction_head().pos in target_pos_list
            and self.in_scope_prediction_head().cue_info_list[neg_num_in_snt] == "_"
            else None
        )

        final_punc: Optional[LeafConstitunent] = (
            self.in_scope_prediction_tail()
            if self.in_scope_prediction_tail().pos in target_pos_list
            and self.in_scope_prediction_tail().cue_info_list[neg_num_in_snt] == "_"
            else None
        )

        initial_or_final_punc: Optional[LeafConstitunent] = initial_punc or final_punc
        return initial_or_final_punc

    def _get_punc_delimited_tree_right_of_cue(
        self,
        target_labels: list[str],
        excluded_subtree: Optional["ParseNode"],
        negation_cue_token_id: int,
    ) -> Optional["ParseNode"]:
        punc_s_ast_right: Optional[ParseNode] = None

        punc_s_ast_right_candidates: list[ParseNode] = [
            subtree
            for subtree in self.subtrees()
            if subtree.label() in target_labels
            and isinstance(subtree, ParseNode)
            and excluded_subtree not in subtree.subtrees()
        ]

        for candidate_tree in punc_s_ast_right_candidates:
            tree_head_token_id: int = candidate_tree.leaf_constituents[0].id
            is_scope_pred_list: list[bool] = [
                leaf_constituent.is_scope_pred
                for leaf_constituent in candidate_tree.leaf_constituents
            ]

            if tree_head_token_id == 0:
                continue
            elif (
                candidate_tree.root().leaf_constituents[tree_head_token_id - 1].pos
                not in PUNCTUATION_LIST
            ):
                continue
            elif negation_cue_token_id >= candidate_tree.leaf_constituents[0].id:
                continue
            elif True not in is_scope_pred_list:
                continue
            else:
                punc_s_ast_right = candidate_tree
                break

        return punc_s_ast_right

    # =============================================================================== #
    # =========================== Post Processing Methods =========================== #
    # =============================================================================== #
    def apply_post_processing(
        self,
        neg_num_in_snt: int,
    ) -> None:
        if self.height() == 2:
            return None

        # get the ParseNode that directly dominates the cue
        negation_cue_tree: ParseNode = self.negation_cue_trees(neg_num_in_snt)[0]
        negation_cue_token_id: int = self.negation_cue_token_ids(neg_num_in_snt)[0]

        # remmove previous conjuncts if the cue is in the conjoined phrase
        self._remove_cc_and_conjuncts_left(neg_num_in_snt, negation_cue_token_id)

        # remove comma-delimited ADJP and INTJ
        self._remove_punctuation_surrounded_parts(
            target_phrases=["ADVP", "INTJ"],
            neg_num_in_snt=neg_num_in_snt,
            excluded_subtree=negation_cue_tree,
            punctuation_list=[",", "."],
        )

    def _remove_cc_and_conjuncts_left(
        self, neg_num_in_snt: int, negation_cue_token_id: int
    ) -> None:
        cc_left: Optional[LeafConstitunent]
        conjuncts_left: Optional[list[ParseNode]]

        cc_left, conjuncts_left = self._get_cc_and_conjuncts_left(negation_cue_token_id)

        if cc_left and conjuncts_left:
            # remove CC
            self.remove_tokens_from_scope_prediction(
                head_token_id=cc_left.id,
                tail_token_id=cc_left.id,
                neg_num_in_snt=neg_num_in_snt,
            )

            # remove conjuncts
            for conjunct_left in conjuncts_left:
                if conjunct_left.height() > 2:
                    head_token_id: int = conjunct_left.leaf_constituents[0].id
                    tail_token_id: int = conjunct_left.leaf_constituents[-1].id
                else:
                    head_token_id: int = conjunct_left.leaf_info.id
                    tail_token_id: int = conjunct_left.leaf_info.id

                self.remove_tokens_from_scope_prediction(
                    head_token_id, tail_token_id, neg_num_in_snt
                )

    def _get_cc_and_conjuncts_left(
        self, negation_cue_token_id: int
    ) -> tuple[Optional[LeafConstitunent], Optional[list["ParseNode"]]]:
        cc_left: Optional[LeafConstitunent] = None
        conjuncts_left: Optional[list[ParseNode]] = None

        tree_pnt: Optional[ParseNode] = (
            self.root().leaf_constituents[negation_cue_token_id].tree
        )
        while tree_pnt != self:
            if tree_pnt.left_siblings() is None:
                tree_pnt = tree_pnt.parent()
                continue

            for left_sib in tree_pnt.left_siblings():
                if left_sib.label() == "CC":
                    cc_left = left_sib.leaf_info
                    break

            if cc_left:
                conjuncts_left = cc_left.tree.left_siblings()

            if cc_left and conjuncts_left:
                break
            else:
                tree_pnt = tree_pnt.parent()

        return cc_left, conjuncts_left

    def _search_and_add_to_list_subtrees_with_certain_parent(
        self,
        tree_label: str,
        parent_label: str,
        target_list: list["ParseNode"],
        is_root_subtree: bool = False,
    ) -> None:
        # add the tree to the list if the height is more than 2 and the conditins are satisfied
        if self.height() > 2:
            if (
                not is_root_subtree
                and self.label() == tree_label
                and self.parent().label() == parent_label
            ):
                target_list.append(self)

            # recursive calls for children
            for child in self:
                child: ParseNode
                child._search_and_add_to_list_subtrees_with_certain_parent(
                    tree_label, parent_label, target_list
                )

    def _remove_punctuation_surrounded_parts(
        self,
        target_phrases: list[str],
        neg_num_in_snt: int,
        excluded_subtree: Optional["ParseNode"] = None,
        punctuation_list: list[str] = PUNCTUATION_LIST,
    ) -> None:
        punc_surrounded_parts: list[ParseNode] = self._get_punctuation_surrounded_parts(
            target_phrases=target_phrases, punctuation_list=punctuation_list
        )

        for tree in punc_surrounded_parts:
            if excluded_subtree in tree.subtrees():
                subtrees_same: list[ParseNode] = [
                    subtree
                    for subtree in tree.subtrees()
                    if subtree.is_same_tree(excluded_subtree)
                ]
                if len(subtrees_same) > 0:
                    continue

            if tree.height() > 2:
                head_token_id: int = tree.leaf_constituents[0].id
                tail_token_id: int = tree.leaf_constituents[-1].id
            else:
                head_token_id: int = tree.leaf_info.id
                tail_token_id: int = tree.leaf_info.id

            self.remove_tokens_from_scope_prediction(
                head_token_id, tail_token_id, neg_num_in_snt
            )

            # remove punctuations
            if (
                self.root().leaf_constituents[head_token_id - 1]
                and self.root().leaf_constituents[head_token_id - 1].pos
                in PUNCTUATION_LIST
            ):
                self.remove_tokens_from_scope_prediction(
                    head_token_id - 1, head_token_id - 1, neg_num_in_snt
                )

            if (
                self.root().leaf_constituents[tail_token_id + 1]
                and self.root().leaf_constituents[tail_token_id + 1].pos
                in PUNCTUATION_LIST
            ):
                self.remove_tokens_from_scope_prediction(
                    tail_token_id + 1, tail_token_id + 1, neg_num_in_snt
                )

    def _get_punctuation_surrounded_parts(
        self, target_phrases: list[str], punctuation_list: list[str]
    ) -> list["ParseNode"]:
        tree_with_target_labels: list[ParseNode] = []
        self._search_and_add_to_list_subtrees(
            target_phrases, tree_with_target_labels, is_root_subtree=True
        )

        punc_surrounded_parts: list[ParseNode] = []
        for tree in tree_with_target_labels:
            head_token_id: int = tree.leaf_constituents[0].id
            tail_token_id: int = tree.leaf_constituents[-1].id

            if (
                head_token_id > 0
                and tail_token_id < len(tree.root().leaf_constituents) - 1
                and tree.root().leaf_constituents[head_token_id - 1].pos
                in punctuation_list
                and tree.root().leaf_constituents[tail_token_id + 1].pos
                in punctuation_list
            ):
                punc_surrounded_parts.append(tree)

        return punc_surrounded_parts

    def _search_and_add_to_list_subtrees(
        self,
        tree_labels: list[str],
        target_list: list["ParseNode"],
        is_root_subtree: bool = False,
    ) -> None:
        if self.height() > 2:
            if not is_root_subtree and self.label() in tree_labels:
                target_list.append(self)

            for child in self:
                child: ParseNode
                child._search_and_add_to_list_subtrees(tree_labels, target_list)

    # =============================================================================== #
    # ========================= Scope Prediction Methods ============================ #
    # =============================================================================== #

    def predict_scopes(
        self,
        apply_modified_rules: bool = False,
    ) -> list[list[str]]:
        """
        Example of the returned value:
            given sentence: I did not go there, because I was unhappy.
            cue1: not -> scope_preidction: I, did, go, there
            cue2: un -> scope_prediction: I was happy
            returned value:
            [
                ['I', 'did', '_', 'go', 'there', '_', '_', '_', '_', '_'],
                ['_', '_', '_', '_', '_', '_', 'I', 'was', 'happy', '_']
            ]
        """
        leaf_constituents: list[LeafConstitunent] | None
        leaf_constituents_in_scope_pred: list[LeafConstitunent] | None
        scope_tree: ParseNode | None

        cue_token_id_list: list[list[int]] = self._cue_token_id_list()

        assert cue_token_id_list is not None, "This sentence has no negation."

        # initialize the list that stores the predictions
        scope_prediction_list: list[list[str]] = [
            [] for _ in range(self.negation_num_in_snt())
        ]

        predicted_tree_info_list: list[dict] = [
            {} for _ in range(self.negation_num_in_snt())
        ]

        for neg_num_in_snt, id_list in enumerate(cue_token_id_list):
            if len(id_list) != 1:
                # apply default scope
                scope_tree = self
                scope_tree.applied_rule_id = 0

                scope_tree.apply_slackening_rules(
                    neg_num_in_snt,
                    initial_and_final_punc_removal=True,
                    apply_modified_rules=apply_modified_rules,
                    not_single_cue_token=True,
                )
                scope_tree.apply_post_processing(
                    neg_num_in_snt,
                )

                leaf_constituents = scope_tree._apply_default_scope_prediction(
                    scope_tree.root(), id_list[0], id_list[-1]
                )
                leaf_constituents_in_scope_pred = (
                    None
                    if leaf_constituents is None
                    else [
                        leaf_constituent
                        for leaf_constituent in leaf_constituents
                        if leaf_constituent.is_scope_pred
                    ]
                )
                scope_prediction_list[neg_num_in_snt] = self._create_scope_predictions(
                    neg_num_in_snt,
                    len(self.leaf_constituents),
                    leaf_constituents_in_sope=leaf_constituents_in_scope_pred,
                )
            else:
                cue_leaf_constituent: LeafConstitunent = self.leaf_constituents[
                    id_list[0]
                ]
                scope_tree = self._apply_heuristics(
                    cue_leaf_constituent, apply_modified_rules=apply_modified_rules
                )

                if scope_tree is None:
                    # apply default scope
                    scope_tree = self
                    scope_tree.applied_rule_id = 0

                    scope_tree.apply_slackening_rules(
                        neg_num_in_snt,
                        initial_and_final_punc_removal=True,
                        apply_modified_rules=apply_modified_rules,
                    )
                    scope_tree.apply_post_processing(
                        neg_num_in_snt,
                    )

                    leaf_constituents = scope_tree._apply_default_scope_prediction(
                        scope_tree.root(), id_list[0], id_list[0]
                    )
                    leaf_constituents_in_scope_pred = (
                        None
                        if leaf_constituents is None
                        else [
                            leaf_constituent
                            for leaf_constituent in leaf_constituents
                            if leaf_constituent.is_scope_pred
                        ]
                    )
                    scope_prediction_list[
                        neg_num_in_snt
                    ] = self._create_scope_predictions(
                        neg_num_in_snt,
                        len(self.leaf_constituents),
                        leaf_constituents_in_sope=leaf_constituents_in_scope_pred,
                    )
                else:
                    # apply slackening rules
                    scope_tree.apply_slackening_rules(
                        neg_num_in_snt,
                        initial_and_final_punc_removal=True,
                        apply_modified_rules=apply_modified_rules,
                    )
                    # apply post processing
                    scope_tree.apply_post_processing(
                        neg_num_in_snt,
                    )
                    leaf_constituents = (
                        None
                        if scope_tree.leaf_constituents is None
                        else scope_tree.leaf_constituents
                    )
                    leaf_constituents_in_scope_pred = (
                        None
                        if leaf_constituents is None
                        else [
                            leaf_constituent
                            for leaf_constituent in leaf_constituents
                            if leaf_constituent.is_scope_pred
                        ]
                    )
                    scope_prediction_list[
                        neg_num_in_snt
                    ] = self._create_scope_predictions(
                        neg_num_in_snt,
                        len(self.leaf_constituents),
                        leaf_constituents_in_sope=leaf_constituents_in_scope_pred,
                    )

            predicted_tree_info_list[neg_num_in_snt] = {
                "applied_rule_id": scope_tree.applied_rule_id,
                "label": scope_tree.label()
                if scope_tree.applied_rule_id != 0
                else "-1",
                "start_token_id": scope_tree.leaf_constituents[0].id
                if scope_tree.leaf_constituents
                else -1,
                "start_token_surface": scope_tree.leaf_constituents[0].surface
                if scope_tree.leaf_constituents
                else "*",
                "end_token_id": scope_tree.leaf_constituents[-1].id
                if scope_tree.leaf_constituents
                else -1,
                "end_token_surface": scope_tree.leaf_constituents[-1].surface
                if scope_tree.leaf_constituents
                else "*",
            }

            # reset prediction list
            self._reset_scope_pred()

        self.predicted_tree_info_list = predicted_tree_info_list
        return scope_prediction_list


class ScopeResolutionHeuristics:
    @classmethod
    def _search_RB_1(
        cls, start: Optional[LeafConstitunent | ParseNode]
    ) -> Optional[ParseNode]:
        """
        ===== heuristics1 =====
        RB//VP/SBAR if SBAR\WH*
        """
        if start is None:
            return None

        # search VP
        vp: Optional[ParseNode] = (
            start.tree.search_ancestors(["VP"])
            if isinstance(start, LeafConstitunent)
            else start.search_ancestors(["VP"])
        )

        # return None if VP is not found
        if vp is None:
            return None

        # recursively call the method if the parent of VP is not SBAR or
        # the SBAR has no children WH*
        sbar: Optional[ParseNode] = vp.search_parent(["SBAR"])
        if (sbar is None) or (sbar.search_children(WH_PHRASE_TAGS) is None):
            cls._search_RB_1(vp)

        elif sbar is not None:
            sbar.applied_rule_id = 1

            return sbar

    @classmethod
    def _search_RB_1_modified_rule(
        cls, start: Optional[LeafConstitunent | ParseNode]
    ) -> Optional[ParseNode]:
        """
        ===== heuristics101 (modified rule) =====
        RB//VP/S/SBAR if SBAR\WHNP
        """
        if start is None:
            return None

        # search VP
        vp: Optional[ParseNode] = (
            start.tree.search_ancestors(["VP"])
            if isinstance(start, LeafConstitunent)
            else start.search_ancestors(["VP"])
        )

        # return None if VP is not found
        if vp is None:
            return None

        s: Optional[ParseNode] = vp.search_parent(["S"])
        if s is None:
            cls._search_RB_1(vp)
        else:
            sbar: Optional[ParseNode] = s.search_parent(["SBAR"])
            if (sbar is None) or (sbar.search_children(["WHNP"]) is None):
                cls._search_RB_1(vp)

            elif sbar is not None:
                sbar.applied_rule_id = 101
                return sbar

    @classmethod
    def _search_RB_2(
        cls, start: Optional[LeafConstitunent | ParseNode]
    ) -> Optional[ParseNode]:
        """
        ===== heuristics2 =====
        RB//VP/S
        """
        if start is None:
            return None

        # search VP
        vp: Optional[ParseNode] = (
            start.tree.search_ancestors(["VP"])
            if isinstance(start, LeafConstitunent)
            else start.search_ancestors(["VP"])
        )

        # return None if VP is not found
        if vp is None:
            return None

        # recursively call the method if the parent of VP is not S
        s: Optional[ParseNode] = vp.search_parent(["S"])
        if s is None:
            cls._search_RB_2(vp)

        if s is not None:
            s.applied_rule_id = 2

        return s

    @classmethod
    def _search_RB_3(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        ===== heuristics3 =====
        RB//S
        """
        s: Optional[ParseNode] = start.tree.search_ancestors(["S"])

        if s is not None:
            s.applied_rule_id = 3

        return s

    @classmethod
    def _search_DT_1(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        ===== heuristics4 =====
        DT/NP if NP/PP
        """
        np: Optional[ParseNode] = start.tree.search_parent(["NP"])
        if (np is None) or (np.search_parent(["PP"]) is None):
            return None
        elif np.search_parent(["PP"]):
            if np is not None:
                np.applied_rule_id = 4

            return np

    @classmethod
    def _search_DT_2(
        cls, start: Optional[LeafConstitunent | ParseNode]
    ) -> Optional[ParseNode]:
        """
        ===== heuristics5 =====
        DT//SBAR if SBAR\WHADVP
        """
        if start is None:
            return None

        # search SBAR
        sbar: Optional[ParseNode] = (
            start.tree.search_ancestors(["SBAR"])
            if isinstance(start, LeafConstitunent)
            else start.search_ancestors(["SBAR"])
        )

        # return None if SBAR is not found
        if sbar is None:
            return None

        # search WHADVP
        whadvp: Optional[ParseNode] = sbar.search_children(["WHADVP"])
        if whadvp is None:
            cls._search_DT_2(sbar)
        else:
            if sbar is not None:
                sbar.applied_rule_id = 5

            return sbar

    @classmethod
    def _search_DT_3(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        ===== heuristics6 =====
        DT//S
        """
        s: Optional[ParseNode] = start.tree.search_ancestors(["S"])

        if s is not None:
            s.applied_rule_id = 6

        return s

    @classmethod
    def _search_JJ_1(
        cls, start: Optional[LeafConstitunent | ParseNode]
    ) -> Optional[ParseNode]:
        """
        ===== heuristics7 =====
        JJ//ADJP/VP/S if S\VP\VB* [@lemma="be"]
        """
        recursive_flag: bool = False

        if start is None:
            return None

        # search ADJP
        adjp: Optional[ParseNode] = (
            start.tree.search_ancestors(["ADJP"])
            if isinstance(start, LeafConstitunent)
            else start.search_ancestors(["ADJP"])
        )

        # return None if ADJP is not found
        if adjp is None:
            return None

        # search VP
        vp: Optional[list[ParseNode]] = adjp.search_parent(["VP"])

        if vp is None:
            recursive_flag = True

        else:
            s: Optional[ParseNode] = vp.search_parent(["S"])
            if s is None:
                recursive_flag = True
            else:
                vp_children_vb_ast: Optional[list[ParseNode]] = vp.search_children(
                    VB_ASTERISK_POS_LIST
                )
                if vp_children_vb_ast is None:
                    recursive_flag = True
                else:
                    for vb_ast in vp_children_vb_ast:
                        if vb_ast.is_leaf and vb_ast.leaf_info.lemma == "be":
                            if s is not None:
                                s.applied_rule_id = 7
                            return s
                    else:
                        recursive_flag = True

        if recursive_flag:
            cls._search_JJ_1(adjp)

    @classmethod
    def _search_JJ_2(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        ===== heuristics8 =====
        JJ/NP/NP if NP\PP
        """
        np: Optional[ParseNode] = start.tree.search_parent(["NP"])
        if np is None:
            return None

        np_upper: Optional[ParseNode] = np.search_parent(["NP"])
        if (np_upper is None) or (np_upper.search_children(["PP"]) is None):
            return None
        elif np_upper.search_children(["PP"]):
            if np_upper is not None:
                np_upper.applied_rule_id = 8

            return np_upper

    @classmethod
    def _search_JJ_3(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        ===== heuristics9 =====
        JJ//NP
        """
        np: Optional[ParseNode] = start.tree.search_ancestors(["NP"])
        if np is not None:
            np.applied_rule_id = 9

        return np

    @classmethod
    def _search_UH(cls, start: LeafConstitunent) -> ParseNode:
        """
        ===== heuristics10 =====
        UH
        """
        if start.tree is not None:
            start.tree.applied_rule_id = 10

        return start.tree

    @classmethod
    def _search_IN(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        ===== heuristics11 =====
        IN//PP
        """
        pp: Optional[ParseNode] = start.tree.search_parent(["PP"])
        if pp is not None:
            pp.applied_rule_id = 11

        return pp

    @classmethod
    def _search_NN_1(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        ===== heuristics12 =====
        NN/NP//S/SBAR if SBAR\WHNP
        """
        np: Optional[ParseNode] = start.tree.search_parent(["NP"])
        if np is None:
            return None

        s: Optional[ParseNode] = np.search_ancestors(["S"])

        while s is not None:
            sbar: Optional[ParseNode] = s.search_parent(["SBAR"])

            if sbar is None:
                s = s.search_ancestors(["S"])
                continue

            assert isinstance(sbar, ParseNode)
            if sbar.search_children(["WHNP"]):
                if sbar is not None:
                    sbar.applied_rule_id = 12

                return sbar
            else:
                s = s.search_ancestors(["S"])
                continue
        else:
            return None

    @classmethod
    def _search_NN_2(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        ===== heuristics13 =====
        NN/NP//S
        """
        np: Optional[ParseNode] = start.tree.search_parent(["NP"])
        if np is None:
            return None

        s: Optional[ParseNode] = np.search_ancestors(["S"])
        if s is not None:
            s.applied_rule_id = 13

        return s

    @classmethod
    def _search_CC(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        ===== heuristics14 =====
        CC/SINV
        """
        sinv: Optional[ParseNode] = start.tree.search_parent(["SINV"])
        if sinv is not None:
            sinv.applied_rule_id = 14

        return sinv

    @classmethod
    def apply_heuristics_to_RB(
        cls, start: LeafConstitunent, apply_modified_rules: bool = False
    ) -> Optional[ParseNode]:
        """
        apply the rules to the negation whose cue is RB
        """
        assert start.pos == "RB"

        # modified rule
        if apply_modified_rules:
            return (
                cls._search_RB_1_modified_rule(start)
                or cls._search_RB_2(start)
                or cls._search_RB_3(start)
            )
        # original rule
        else:
            return (
                cls._search_RB_1(start)
                or cls._search_RB_2(start)
                or cls._search_RB_3(start)
            )

    @classmethod
    def apply_heuristics_to_DT(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        apply the rules to the negation whose cue is DT
        """
        assert start.pos == "DT"

        return (
            cls._search_DT_1(start)
            or cls._search_DT_2(start)
            or cls._search_DT_3(start)
        )

    @classmethod
    def apply_heuristics_to_JJ(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        apply the rules to the negation whose cue is JJ
        """
        assert start.pos == "JJ"

        return (
            cls._search_JJ_1(start)
            or cls._search_JJ_2(start)
            or cls._search_JJ_3(start)
        )

    @classmethod
    def apply_heuristics_to_UH(cls, start: LeafConstitunent) -> ParseNode:
        """
        apply the rules to the negation whose cue is UH
        """
        assert start.pos == "UH"

        return cls._search_UH(start)

    @classmethod
    def apply_heuristics_to_IN(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        apply the rules to the negation whose cue is IN
        """
        assert start.pos == "IN"

        return cls._search_IN(start)

    @classmethod
    def apply_heuristics_to_NN(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        apply the rules to the negation whose cue is NN
        """
        assert start.pos == "NN"

        return cls._search_NN_1(start) or cls._search_NN_2(start)

    @classmethod
    def apply_heuristics_to_CC(cls, start: LeafConstitunent) -> Optional[ParseNode]:
        """
        apply the rules to the negation whose cue is CC
        """
        assert start.pos == "CC"

        return cls._search_CC(start)
