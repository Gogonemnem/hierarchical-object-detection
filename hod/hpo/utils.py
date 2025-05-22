def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, key, sep))
        else:
            items[key] = v
    return items


def unflatten_dict(flat, sep="."):
    nested = {}
    for k, v in flat.items():
        parts = k.split(sep)
        current = nested
        for p in parts[:-1]:
            current = current.setdefault(p, {})
        current[parts[-1]] = v
    return nested
