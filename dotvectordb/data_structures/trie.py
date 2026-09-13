
class TrieNode:
    """
    TrieNode represents a single node in the trie.
    children: It contains a dictionary where each key is an item in the path
    is_terminal: This flag indicates whether the node represents the end of a valid path.
    self.value: The value is the payload associated with a completed path. 
    """
    def __init__(self):
        """Each node has a dictionary of children, a boolean indicating if it's terminal, and an optional value."""
        self.children = {}
        self.is_terminal = False
        self.value = None
    def validate(self):
        """
        Ensure only terminal nodes contain a value.
        Raises:
            ValueError: If a non-terminal node contains a value.
        """
        if not self.is_terminal and self.value is not None:
            raise ValueError(
                "Invalid TrieNode state: non-terminal nodes cannot contain a value."
            )
        return True

class PathTrie:
    """
    Trie that keeps only maximal paths.
    In DotVectorDB, there's no point to keep [1,2] if [1,2,3] is already stored.
    It does not carry any new innformation, so it is discarded.
    Rules:
        Case 1: Subset of an already stored path is rejected.
        existing: (1, 2, 3)
        add:      (1, 2)
        -> rejected because candidate is shorter

        Case 2: Superset of an already stored path replaces it.
        existing: (1, 2)
        add:      (1, 2, 3)
        -> existing shorter path is replaced

        Case 3: Neither is a prefix of the other, both are kept.
        existing: (1, 2, 3)
        add:      (1, 4)
        -> both remain because neither is a prefix of the other
    """
    def __init__(self):
        """The root node is a dummy node that does not represent any item in the path."""
        self.root = TrieNode()

    def add(self, path, value=None):
        """This method adds a new path to the trie, ensuring that only maximal paths are kept
        path: A list or tuple representing the path to be added. This is used to navigate the trie.
        value: If we navigate to a terminal node, this value will be stored with the path.
        Returns:
        True if the path was added, False otherwise.
        """

        # First, we convert the path to a tuple to ensure it's hashable and can be used as a key in the children dictionary.
        path = tuple(path)
        if not path:
            return False, None

        #####################################################
        # Cases When We are at an intermediate node         #
        #####################################################    
        # We copy the dummy root node to start traversing the trie.
        node = self.root
        old_value = None
        for item in path:
            # Case 1: If we encounter a terminal node while traversing, it means the new path is a superset of an existing path.
            # Solution: We replace the existing shorter path with the new longer path by marking the existing node as non-terminal and clearing its value.
            if node.is_terminal:
                node.is_terminal = False
                old_value = node.value
                node.value = None

            # Case 2: If we encounter a non-terminal node, we continue traversing.
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]

        #####################################################
        # Cases When We Reach the End of the New Path       #
        #####################################################    
        # Case 3 (CANNOT ADD): If we reach the end of the new path and the node is already terminal, it means the exact path already exists in the trie.
        if node.is_terminal:
            return False, None

        # Case 4 (CANNOT ADD): If we reach the end of the new path and the node has children, it means the new path is a subset of an existing path.
        if node.children:
            return False, None

        # Case 5 (CAN ADD): If we reach the end of the new path and the node is not terminal and has no children, it means the new path is unique and can be added to the trie. We replace the existing shorter path with the new longer path by marking the node as terminal and storing the value.
        node.is_terminal = True
        node.value = path if value is None else value
        return True, old_value

    def contains_exact(self, path):
        node = self._find_node(path)
        return node is not None and node.is_terminal

    def contains_prefix(self, path):
        return self._find_node(path) is not None

    def _find_node(self, path):
        node = self.root
        for item in path:
            node = node.children.get(item)
            if node is None:
                return None
        return node

    def __contains__(self, path):
        return self.contains_prefix(path)

    def values(self):
        """Return values for all currently surviving terminal paths."""
        results = []

        def walk(node):
            if node.is_terminal:
                results.append(node.value)

            for child in node.children.values():
                walk(child)

        walk(self.root)
        return results

    def paths(self):
        """Return all currently surviving paths."""
        results = []
        def walk(node, current):
            if node.is_terminal:
                results.append(tuple(current))
            for item, child in node.children.items():
                current.append(item)
                walk(child, current)
                current.pop()
        walk(self.root, [])
        return results

    def display(self):
        print("ROOT")

        def walk(node, depth):
            for item, child in node.children.items():
                marker = " [TERMINAL]" if child.is_terminal else ""
                print(f"{'  ' * depth}{item}{marker}")
                walk(child, depth + 1)

        walk(self.root, 0)