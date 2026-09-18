
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
    Trie that stores distinct paths without treating prefixes as duplicates.
    A path is rejected only when the same relationship sequence, or the full
    reverse of that sequence, is already stored.
    """
    def __init__(self):
        """The root node is a dummy node that does not represent any item in the path."""
        self.root = TrieNode()

    def add(self, path, value=None):
        """Add a path unless the same path or its reverse is already stored.
        path: A list or tuple representing the path to be added. This is used to navigate the trie.
        value: If we navigate to a terminal node, this value will be stored with the path.
        Returns:
        True if the path was added, False otherwise.
        """

        # First, we convert the path to a tuple to ensure it's hashable and can be used as a key in the children dictionary.
        path = tuple(path)
        if not path:
            return False, None

        if self.contains_exact(path) or self.contains_exact(reversed(path)):
            return False, None

        node = self.root
        for item in path:
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]

        node.is_terminal = True
        node.value = path if value is None else value
        return True, None

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
        path = tuple(path)
        return (
            self.contains_exact(path)
            or self.contains_exact(reversed(path))
        )

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
