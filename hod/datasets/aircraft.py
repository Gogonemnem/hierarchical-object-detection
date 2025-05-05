from mmengine.fileio import load
from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module()
class AircraftDataset(CocoDataset): # Change to HierarchicalDataset?
    """Dataset for Aircraft."""
    METAINFO = {
        **CocoDataset.METAINFO,
        'taxonomy' : None
    }
    def load_data_list(self):
        ann = load(self.ann_file)
        cats = ann['categories']
        # store the list of class-names (by COCO id order):
        self._metainfo['classes'] = [c['name'] for c in sorted(cats, key=lambda c: c['id'])]
        # pass along the tree too:
        self._metainfo['taxonomy'] = ann.get('taxonomy', {})
        data_list = super().load_data_list()
        # also build name->id map:
        self._metainfo['class_to_idx'] = {
            name: self.cat2label[coco_id]
            for coco_id, name in zip(self.cat_ids, self.metainfo['classes'])
        }

        import matplotlib.pyplot as plt

        def generate_palette(n):
            cmap = plt.get_cmap("tab20")  # 20-color map for high contrast
            colors = []
            for i in range(n):
                color = cmap(i % 20)  # cycle if more than 20
                rgb = tuple(int(255 * c) for c in color[:3])
                colors.append(rgb)
            return colors
        self._metainfo['palette'] = generate_palette(len(self._metainfo['classes']))
        return data_list
