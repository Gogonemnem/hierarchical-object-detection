from torch import nn
from mmengine.fileio import load
from hod.utils.tree import HierarchyTree

class HierarchicalDataMixin:
    def __init__(self, ann_file, **kwargs):
        self.ann_file = ann_file
        self.load_taxonomy(self.ann_file)
    
    def load_taxonomy(self, ann_file):
        ann = load(ann_file)
        taxonomy = ann.get('taxonomy', {})
        self.tree = HierarchyTree(taxonomy)
        self.class_to_idx = {c['name']: c['id'] for c in ann['categories']}
        self.idx_to_class = {c['id']: c['name'] for c in ann['categories']}
