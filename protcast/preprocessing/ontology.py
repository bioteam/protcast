from __future__ import annotations
from protcast import BP, CC, MF
import logging
import pickle
from tqdm import tqdm
from typeguard import typechecked


class GOTerm:
    """GOTerm
    This class represents a GO term

    Attributes
    ----------
    id: str
        ....
    namespace: str
        GO namespace
    name: str
        ....
    primary: Boolean
        A primary term is the most specific and detailed description of gene or
        gene product functions. Usually found in the lowest level of the tree.
    is_obsolete: Boolean
        ....
    level: int
        Level in ontology tree, starting at level 0 (Molecular Function,
        Biological Process, Cellular Component)
    parents: dict
        Key is id, value is GOTerm
    children:
        ...
    ancestors:
        ...
    annotations: list
        ...

    Methods
    -------
    init:
        Initialize
    add_parent:
        ...
    get_parents:
        ...
    add_child:
        ...
    get_children:
        ...
    get_primary:
        ...
    is_primary:
        ...
    set_level:
        ...
    __lt__:
        ...
    __gt__:
        ...
    __eq__:
        ...
    to_text:
        ...
    """

    @typechecked
    def __init__(
        self,
        id: str,
        namespace: str,
        name: str | None,
        primary: GOTerm | None,  # Boolean below, not GOTerm
        level: int | None,
        is_obsolete: bool,
    ) -> None:
        """__init__

        Parameters
        ----------
        id: str
            GO term id
        namespace: str
            GO namespace
        name: str
            GO term name
        primary: Boolean
            Is primary or not
        level: int
            GO tree level
        is_obsolete: Boolean
            Default: false

        Returns
        -------
        None
        """
        self.id = id
        self.namespace = namespace
        self.name = name
        self.primary = primary
        self.is_obsolete = is_obsolete
        self.parents = {}
        self.children = {}
        self.annotations = []
        self.level = level
        self.ancestors: list[str] = None

    def add_parent(self, parent: GOTerm) -> None:
        """add_parent
        Add parent id to term

        Parameters
        ----------
        parent: Object
            GOTerm

        Returns
        -------
        None
        """
        self.parents[parent.id] = parent

    def get_parents(self) -> list[GOTerm]:
        """get_parents
        Get parents of GOTerm

        Parameters
        ----------
        None

        Returns
        -------
        list of GOTerms
        """
        return self.parents.values()

    def add_child(self, child: GOTerm) -> None:
        """add_child
        Add id of child to GOTerm

        Parameters
        ----------
        child: object
            GOTerm

        Returns
        -------
        None
        """
        self.children[child.id] = child

    def get_children(self) -> list[GOTerm]:
        """get_children
        Get children of GOTerm

        Parameters
        ----------
        None

        Returns
        -------
        list of GOTerms
        """
        return self.children.values

    def get_primary(self) -> GOTerm:
        """get_primary
        Get primary GOTerm, if primary is None then this node is primary

        Parameters
        ----------
        None

        Returns
        -------
        GOTerm
        """
        if self.primary:
            return self.primary
        else:
            return self

    def is_primary(self) -> bool:
        """is_primary
        Return True, False

        Parameters
        ----------
        None

        Returns
        -------
        Boolean
        """
        return self.primary is None

    def set_level(self, level: int) -> None:
        """set_level
        Set level of GOTerm

        Parameters
        ----------
        level: int
            Level of GOTerm in tree

        Returns
        -------
        None
        """
        self.level = level

    def __lt__(self, obj: GOTerm):
        """__lt__
        Compare GOTerm to self, return True if input GOTerm has higher
        level number, False if input GOTerm has lower level number.
        If the two have the same level then return True if input GOTerm
        has fewer manual, non-obsolete Annotations, else False.

        Parameters
        ----------
        obj: GOTerm
            GOTerm

        Returns
        -------
        Boolean
        """
        if self.level < obj.level:
            return True
        elif self.level > obj.level:
            return False
        else:
            if len(self.get_manual_non_obsolete_annotations()) > len(
                obj.get_manual_non_obsolete_annotations()
            ):
                return True
            else:
                return False

    def __gt__(self, obj: GOTerm):
        """__gt__
        Compare GOTerm to self, return True if input GOTerm has lower
        level number, False if input GOTerm has higher level number.
        If the two have the same level then return True if input GOTerm
        has more manual, non-obsolete Annotations, else False.

        Parameters
        ----------
        obj: GOTerm
            GOTerm

        Returns
        -------
        Boolean
        """
        if self.level > obj.level:
            return True
        elif self.level < obj.level:
            return False
        else:
            if len(self.get_manual_non_obsolete_annotations()) < len(
                obj.get_manual_non_obsolete_annotations()
            ):
                return True
            else:
                return False

    def __eq__(self, obj: GOTerm):
        """__eq__
        Compare input GOTerm to self, see if they are identical

        Parameters
        ----------
        obj: GOTerm
            GOT

        Returns
        -------
        Boolean
        """
        return self is obj

    def to_text(self) -> str:
        """to_text
        Return text summary of GOTerm, including number of Annotations
        number of manual Annotations, id, level, and ids of manual
        Annotations, and whether the Annotations are obsolete

        Parameters
        ----------
        None

        Returns
        -------
        Str
        """
        annots = len(self.get_non_obsolete_annotations())
        manual_annots = len(self.get_manual_non_obsolete_annotations())

        return (
            f"{self.id}\t{self.level}\t{manual_annots}\t"
            f"{annots - manual_annots}\t{annots}\t{self.is_obsolete}\t"
            f"{self.get_primary().id}"
        )


class GODAG:
    """GODAG
    This class represents an acyclic tree of GOTerms

    Attributes
    ----------
    name: str
        Name of tree (BP, MF, CC)
    nodes: dict
        Keys are ids, values are GOTerms

    Methods
    -------
    init:
        Initialize
    add_term:
        ...
    get_term:
        ...
    populate_node_ancestors:
        ...
    populate_node_ancestors_inner:
        ...
    populate_node_levels:
        ...
    populate_node_levels_inner:
        ...
    to_text:
        ...
    to_files:
        ...
    get_valid_terms:
        ...
    get_obsolete_nodes:
        ...
    """

    def __init__(self, name) -> None:
        """__init__
        Initialize

        Parameters
        ----------
        name: str
            MF, BP, or CC
        nodes: dict
            Keys are ids, values are GOTerms

        Returns
        -------
        None
        """
        self.name = name
        self.nodes: dict[str, GOTerm] = {}

    def add_term(self, node: GOTerm):
        """add_term
        Add node to nodes dict where key is id and value is GOTerm

        Parameters
        ----------
        node: obj
            GOTerm

        Returns
        -------
        None
        """
        self.nodes[node.id] = node

    @typechecked
    def get_term(self, node_id: str):
        """get_term
        Get GOTerm given an id

        Parameters
        ----------
        node_id: str
            Id of GOTerm

        Returns
        -------
        GOTerm
        """
        return self.nodes.get(node_id)

    def populate_node_ancestors(self) -> None:
        """populate_node_ancestors
        Set ancestors of self to a list of ids

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logging.debug(f"Populating node ancestors of {self.name}")
        for node in self.nodes.values():
            if node.ancestors is None:
                self.populate_node_ancestors_inner(node)

    def populate_node_ancestors_inner(self, node: GOTerm) -> list[str]:
        """populate_node_ancestors_inner
        Recursive method that gets ids of all ancestors (parents)
        of given node.

        Parameters
        ----------
        node: GOTerm
            GOTerm

        Returns
        -------
        List of ids
        """
        ancestors = set()
        for parent in node.get_parents():
            parent_ancestors = parent.ancestors
            if parent_ancestors is None:
                parent_ancestors = self.populate_node_ancestors_inner(parent)

            ancestors.update(parent_ancestors)
            ancestors.add(parent.id)

        node.ancestors = list(ancestors)

        return node.ancestors

    def populate_node_levels(self) -> None:
        """populate_node_levels
        ...

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logging.debug(f"Populating node levels of {self.name}")
        stack = []
        for node in self.nodes.values():
            if node.is_primary():
                if not node.level:
                    self.populate_node_levels_inner(node, stack)
            else:
                if not node.level:
                    primary_node = node.get_primary()
                    if not primary_node.level:
                        self.populate_node_levels_inner(primary_node, stack)
                    node.set_level(primary_node.level)
            assert not stack

    def populate_node_levels_inner(
        self, node: GOTerm, stack: list[GOTerm]
    ) -> int:
        """populate_node_levels_inner
        Recursive method that sets the level of input GOTerm and
        returns the level of the GOTerm

        Parameters
        ----------
        node: obj
            GOTerm
        stack: list
            List of GOTerms

        Returns
        -------
        Int
        """
        if node in stack:
            print(f"Found a cycle with term: {node.id} in tree: {self.name}")
            print(f"Stack: {[x.id for x in stack]}")
            exit(1)

        stack.append(node)

        if not node.get_parents():
            node_level = 0  # Root node
        else:
            min_parent_level = float("inf")
            for parent in node.get_parents():
                parent_level = parent.level
                if parent.level is None:
                    parent_level = self.populate_node_levels_inner(
                        parent, stack
                    )

                min_parent_level = min(min_parent_level, parent_level)
            node_level = min_parent_level + 1

        node.set_level(node_level)

        stack.remove(node)
        return node_level

    def to_text(self) -> str:
        """to_text
        Write tab-delimited data for each node: term, level and all
        details on manual, electronic, number of annotations, obsolete
        and primary id.

        Parameters
        ----------
        None

        Returns
        -------
        Str
        """
        tree_str = (
            "Term\tLevel\tManual_Annots\tElectronic_Annots\tTotal_Annots\t"
            "Is_obsolete\tPrimary_ID\n"
        )
        tree_str += "\n".join(
            [
                node.to_text()
                for node in sorted(
                    filter(
                        lambda x: not x.is_obsolete,
                        self.nodes.values(),
                    )
                )
            ]
        )
        tree_str += "\n"
        return tree_str

    def to_files(self, output_path):
        """to_files
        Write "term" and "rel" TSV files for all primary and
        non-obsolete GOTerms. The "term" file has term ids and
        names and the "rel" file has term id and parent id.

        Parameters
        ----------
        output: Path
            Location of output *tsv files

        Returns
        -------
        None
        """
        term_output_path = output_path + f"_{self.name}_term.tsv"
        relationship_output_path = output_path + f"_{self.name}_rel.tsv"

        with open(term_output_path, "w") as term_file:
            for term in self.nodes.values():
                if not term.is_obsolete and term.is_primary():
                    term_file.write(term.id + "\t" + term.name + "\n")

        with open(relationship_output_path, "w") as rel_file:
            for term in self.nodes.values():
                if not term.is_obsolete and term.is_primary():
                    for parent in term.get_parents():
                        rel_file.write(term.id + "\tis_a\t" + parent.id + "\n")

    @typechecked
    def get_valid_terms(self) -> list[GOTerm]:
        """get_valid_terms
        Get list of valid GOTerms where valid GOTerms are defined as being
        not obsolete and principal (not alt_id)

        Parameters
        ----------
        None

        Returns
        -------
        List of GOTerms
        """
        return [
            node
            for node in sorted(
                filter(
                    lambda x: not x.is_obsolete and x.is_primary(),
                    self.nodes.values(),
                )
            )
        ]

    def get_obsolete_nodes(self) -> list[GOTerm]:
        """get_obsolete_nodes
        Get obsolete GOTerms

        Parameters
        ----------
        None

        Returns
        -------
        List of GOTerms
        """
        return sorted(filter(lambda x: x.is_obsolete, self.nodes.values()))


class Ontology:
    """Ontology
    This class reads input files and creates a GODAG for each of
    the GO namespaces.

    Attributes
    ----------
    bp_dag: GODAG
        GODAG(BP)
    cc_dag: GODAG
        GODAG(CC)
    mf_dag: GODAG
        GODAG(MF)
    terms: dict
        Keys are term ids and values are GOTerms

    Methods
    -------
    init:
        Initialize
    process_go_term:
        ...
    add_term:
        ...
    get_term:
        ...
    get_primary_term:
        ...
    save:
        ...
    to_files:
        ...
    populate_ontology_levels:
        ...
    populate_ontology_ancestry:
        ...
    load_ontology:
        ...
    """

    def __init__(self, go_file: str) -> None:
        """__init__
        Initialize, read input GO file, and create GODAG objects. Asserts
        that there are no GO terms whose parents are alternate nodes.

        Parameters
        ----------
        input_file: str
            Name of GO ontology file

        Returns
        -------
        None
        """
        self.bp_dag = GODAG(BP)
        self.cc_dag = GODAG(CC)
        self.mf_dag = GODAG(MF)
        self.terms: dict[str, GOTerm] = {}

        with open(go_file, "r") as f:
            lines = [x.strip("\n") for x in f.readlines()]

        goterm_lines = []

        for line in tqdm(lines, desc=f"Processing GO terms from '{go_file}'"):
            # Look for end of Term
            if line == "":
                self.process_go_term(goterm_lines)
                goterm_lines = []
            else:
                goterm_lines.append(line)
        self.process_go_term(goterm_lines)

        self.populate_ontology_levels()
        self.populate_ontology_ancestry()

        logging.info(
            f"Found {len(self.terms.values())} total GO ids "
            f"and alt_ids in '{go_file}'"
        )

        for term in self.terms.values():
            assert term.name
            for parent in term.get_parents():
                assert parent.is_primary()

    def process_go_term(self, goterm_lines: list(str)):
        """process_go_term
        Gets id, name, namespace, alt_id, is_a (parent), and
        is_obsolete from different lines and builds GOTerms from
        these values.

        Parameters
        ----------
        goterm_lines: list
            List of lines from GO file

        Returns
        -------
        None
        """
        if not goterm_lines:
            return
        num_go_terms = 0
        header = goterm_lines[0]

        if header != "[Term]":
            logging.debug(f"First line is {header}. Skipping ...")
            return

        parents = []
        alt_go_nodes = []
        is_obsolete = False
        for line in goterm_lines:
            if line.startswith("id:"):
                term_id = line.split(":", 1)[1].strip()
                num_go_terms += 1
                continue
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
                continue
            if line.startswith("namespace:"):
                namespace = line.split(":", 1)[1].strip()
                continue
            if line.startswith("alt_id:"):
                alt_go_node = line.split(":", 1)[1].strip()
                alt_go_nodes.append(alt_go_node)
                continue
            if line.startswith("is_a:"):
                parent = line.split(":", 1)[1].split("!")[0].strip()
                parents.append(parent)
                continue
            if line.startswith("is_obsolete:"):
                is_obsolete = True
                continue

        assert term_id is not None
        assert namespace
        assert name
        assert not (
            is_obsolete and parents
        )  # Terms that are obsolete should not have/be parents

        go_node = self.get_term(term_id)
        if go_node is None:
            go_node = GOTerm(term_id, namespace, name, None, None, is_obsolete)
        else:
            assert not is_obsolete  # A parent node cannot be obsolete
            go_node.is_obsolete = (
                is_obsolete  # For cases where the Term was created as a parent
            )
            go_node.name = name

        for parent in parents:
            parent_node = self.get_term(parent)
            if not parent_node:
                parent_node = GOTerm(
                    parent, namespace, None, None, None, is_obsolete
                )
            go_node.add_parent(parent_node)
            parent_node.add_child(go_node)
            self.add_term(parent_node)

        for alt_go_node_id in alt_go_nodes:
            # Confirm that the alt_id is not another node in the Ontology
            # If happens that means that it is the parent of some node?
            assert self.get_term(alt_go_node_id) is None
            alt_go_node = GOTerm(
                alt_go_node_id,
                namespace,
                name,
                None,
                None,
                is_obsolete,
            )
            self.add_term(alt_go_node)

        self.add_term(go_node)

    def add_term(self, go_node: GOTerm):
        """add_term
        Add GOTerm to specific GODAG based on namespace

        Parameters
        ----------
        go_node: GOTerm
            GO term

        Returns
        -------
        None
        """
        self.terms[go_node.id] = go_node
        if go_node.namespace == BP:
            self.bp_dag.add_term(go_node)
        elif go_node.namespace == MF:
            self.mf_dag.add_term(go_node)
        elif go_node.namespace == CC:
            self.cc_dag.add_term(go_node)
        else:
            logging.error(
                "Trying to add a GO term to an unrecognized namespace: "
                f"{go_node.namespace}"
            )
            exit(1)

    @typechecked
    def get_term(self, term_id: str) -> GOTerm | None:
        """get_terms
        Return GOTerm given a GO term id or None

        Parameters
        ----------
        term_id: str
            Id of GO term

        Returns
        -------
        GOTerm or None
        """
        return self.terms.get(term_id)

    @typechecked
    def get_primary_term(self, node_id: str) -> GOTerm | None:
        """get_primary_term
        Get primary GOTerm or None given a GO term id

        Parameters
        ----------
        node_id: str
            GO term id

        Returns
        -------
        GOTerm or None
        """
        go_term = self.terms.get(node_id)
        if go_term is not None:
            return go_term.get_primary()
        else:
            return None

    def save(self, output_file: str) -> str:
        """save
        Serialize an ontology

        Parameters
        ----------
        output_file: str
            Name of output file

        Returns
        -------
        File name
        """
        with open(output_file, "wb") as f:
            pickle.dump(self, f)
        return output_file

    def to_files(self, output_path: str):
        """to_files
        Call GODAG.to_files to serialize the three namespace GODAGs

        Parameters
        ----------
        output_path: str
            Name of output directory

        Returns
        -------
        None
        """
        self.bp_dag.to_files(output_path)
        self.cc_dag.to_files(output_path)
        self.mf_dag.to_files(output_path)

    def populate_ontology_levels(self) -> None:
        """populate_ontology_levels
        Call GODAG.populate_node_levels for each namespace

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.bp_dag.populate_node_levels()
        self.cc_dag.populate_node_levels()
        self.mf_dag.populate_node_levels()

    def populate_ontology_ancestry(self) -> None:
        """populate_ontology_ancestry
        Call GODAG.populate_node_ancestors for each namespace

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.bp_dag.populate_node_ancestors()
        self.cc_dag.populate_node_ancestors()
        self.mf_dag.populate_node_ancestors()

    @classmethod
    def load_ontology(cls, input_file: str) -> Ontology:
        """load_ontology
        Read serialized ontology file

        Parameters
        ----------
        input_file: str
            Name of serialized ontology file

        Returns
        -------
        Ontology
        """
        with open(input_file, "rb") as f:
            ontology = pickle.load(f)

        return ontology
