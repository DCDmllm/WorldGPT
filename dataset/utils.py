import torch


def to(data, device=None, dtype=None):
    for m in data:
        if isinstance(data[m], torch.Tensor):
            if dtype:
                data[m] = data[m].to(dtype=dtype)
            if device:
                data[m] = data[m].to(device)
    return data


def mask_modality(data, return_modality):
    return_data = {}
    for m in data:
        if m in return_modality or m == 'text':
            return_data[m] = data[m]
    return return_data

def serialize_task(task):
    serialized = '('
    for i, m in enumerate(task['inputs']):
        if i > 0:
            serialized += ', '
        serialized += m
    serialized += ') - ('
    for i, m in enumerate(task['targets']):
        if i > 0:
            serialized += ', '
        serialized += m
    serialized += ')'
    return serialized