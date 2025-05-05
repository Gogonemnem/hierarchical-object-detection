from typing import List, Dict, Union, Optional, Any

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
    
    def __repr__(self) -> str:
        return self.name


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

    def get_depth(self, cls_name: str) -> int:
        return len(self.get_ancestors(cls_name))

    def get_path(self, cls_name: str) -> List[str]:
        return self.get_ancestors(cls_name) + [cls_name]

    def all_classes(self) -> List[str]:
        return list(self.class_to_node.keys())
    
    def max_depth(self) -> int:
        depths = []
        for name, node in self.class_to_node.items():
            if node.is_leaf():
                depths.append(self.get_depth(name))
        return max(depths)

