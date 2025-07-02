from typing import List, Dict, Optional, Any

class HierarchyNode:
    def __init__(self, name: str, children: Optional[List['HierarchyNode']] = None, 
                 parent: Optional['HierarchyNode'] = None) -> None:
        self.name = name
        self.children = children or []
        self.parent = parent

        for child in self.children:
            child.parent = self

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def ancestors(self) -> List[str]:
        node, path = self, []
        while node.parent:
            node = node.parent
            path.append(node.name)
        return path[::-1]

    def descendants(self) -> List[str]:
        """
        Returns a list of all descendant class names.
        """
        descendants = []
        stack: List['HierarchyNode'] = [self]
        while stack:
            node = stack.pop()
            descendants.append(node.name)
            stack.extend(node.children)
        return descendants

    def get_depth(self) -> int:
        """
        Returns the depth of the class in the hierarchy.
        The depth is defined as the number of edges from the root to the class.
        """
        return len(self.ancestors())

    def get_height(self) -> int:
        """
        Returns the height of the subtree rooted at this node.
        The height is defined as the number of edges from the node to the deepest leaf.
        """
        if self.is_leaf():
            return 0
        return 1 + max(child.get_height() for child in self.children)

    def __repr__(self) -> str:
        return self.name
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HierarchyNode):
            return False
        return self.name == other.name and self.children == other.children
    def __lt__(self, other: 'HierarchyNode') -> bool:
        return self.name < other.name
    def __hash__(self) -> int:
        return hash(self.name) ^ hash(tuple(self.children))


class HierarchyTree:
    def __init__(self, taxonomy_dict: Dict[str, Any]) -> None:
        if len(taxonomy_dict) != 1:
            raise ValueError("Taxonomy dict must have exactly one root node")

        root_name, root_subtree = next(iter(taxonomy_dict.items()))
        self.root = self._build_tree(root_name, root_subtree)
        self.class_to_node: Dict[str, HierarchyNode] = {}
        self._register_nodes(self.root)

    def _build_tree(self, name: str, subtree: Dict[str, Any]) -> HierarchyNode:
        children = [self._build_tree(child_name, child_subtree)
                    for child_name, child_subtree in subtree.items()]
        return HierarchyNode(name, children)

    def _register_nodes(self, node: HierarchyNode) -> None:
        self.class_to_node[node.name] = node            
        for child in node.children:
            self._register_nodes(child)

    def get_ancestors(self, cls_name: str) -> List[str]:
        return self.class_to_node[cls_name].ancestors()

    def get_descendants(self, cls_name: str) -> List[str]:
        return self.class_to_node[cls_name].descendants()

    def get_path(self, cls_name: str) -> List[str]:
        return self.get_ancestors(cls_name) + [cls_name]

    def is_descendant(self, node_name: str, ancestor_name: str) -> bool:
        """Check if a node is a descendant of another node."""
        if node_name not in self.class_to_node or ancestor_name not in self.class_to_node:
            return False
        # A node is considered a descendant of itself for this logic.
        if node_name == ancestor_name:
            return True
        
        path_to_node = self.get_path(node_name)
        return ancestor_name in path_to_node

    def all_classes(self) -> List[str]:
        return list(self.class_to_node.keys())

    def get_siblings(self, cls_name: str) -> List[str]:
        """Get sibling class names (same parent, excluding itself)."""
        node = self.class_to_node[cls_name]
        if node.parent is None:
            return []  # Root has no siblings
        return [child.name for child in node.parent.children if child.name != cls_name]
    
    def get_grandparent(self, cls_name: str) -> str | None:
        """Get grandparent class name (parent's parent)."""
        node = self.class_to_node[cls_name]
        if node.parent is None or node.parent.parent is None:
            return None  # Root or children of root have no grandparent
        return node.parent.parent.name

    def get_cousins(self, cls_name: str) -> List[str]:
        """Get cousin class names (children of parent's siblings) and parent's siblings themselves."""
        node = self.class_to_node[cls_name]
        if node.parent is None or node.parent.parent is None:
            return []  # Root or children of root have no cousins
        
        cousins = []
        # Get parent's siblings
        parent_siblings = [child for child in node.parent.parent.children 
                          if child.name != node.parent.name]
        
        # Include parent's siblings themselves (cousin parents)
        cousins.extend([sibling.name for sibling in parent_siblings])
        
        # Get all children of parent's siblings (traditional cousins)
        for sibling in parent_siblings:
            cousins.extend([child.name for child in sibling.children])
        return cousins

    def get_leaf_nodes(self) -> List[HierarchyNode]:
        return [node for node in self.class_to_node.values() if node.is_leaf()]
    
    def __len__(self) -> int:
        return len(self.class_to_node)