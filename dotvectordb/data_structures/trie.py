"""
Author: Larnell Moore
Purpose:
"""
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_terminal = False
class PathTrie:
    def __init__(self):
        self.root = TrieNode()
    def add(self, path):
        """Add a path to the trie."""
        node = self.root
        for item in path:
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
        node.is_terminal = True
    def contains_prefix(self, path):
        """
        Return True if `path` is a prefix of any path already stored.
        Example:
            stored: [1, 2, 3]
            query:  [1, 2]      -> True
            query:  [1, 2, 3]   -> True
            query:  [1, 2, 4]   -> False
        """
        node = self.root
        for item in path:
            if item not in node.children:
                return False
            node = node.children[item]
        return True
    def contains_exact(self, path):
        """Return True only if the exact path was previously added.
        This candidate is already covered by a longer/equal stored path"""
        node = self.root

        for item in path:
            if item not in node.children:
                return False

            node = node.children[item]

        return node.is_terminal
    def __contains__(self, path):
        return self.contains_prefix(path)