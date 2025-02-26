from mmengine.utils import is_str
from mmdet.evaluation import dataset_aliases

def dior_classes() -> list:
    """Class names of DIOR."""
    return [
        "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
        "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station",
        "groundtrackfield", "golffield", "harbor", "overpass", "ship", "stadium", 
        "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"
    ]

dataset_aliases.update({
    'dior': ['dior', 'DIOR', 'DIOR-VOC']
})


def get_classes(dataset) -> list:
    """Get class names of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if is_str(dataset):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + '_classes()')
        else:
            raise ValueError(f'Unrecognized dataset: {dataset}')
    else:
        raise TypeError(f'dataset must a str, but got {type(dataset)}')
    return labels