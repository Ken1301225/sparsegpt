import torch
import torch.nn as nn


# DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    """_summary_

    Args:
        module (_type_): _description_
        layers (list, optional): _description_. Defaults to [nn.Conv2d, nn.Linear].
        name (str, optional): _description_. Defaults to ''.

    Returns:
        返回当前模块树中所有匹配的模块,返回的是在设定的layers里的模块
    """    
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_MoE_layers(module, layers=[nn.Linear], name='', model_name = 'Qwen1.5'):
    if 'Qwen' in model_name:
        if type(module) in layers and 'experts' in name:
            return {name: module}
        res = {}
        for name1, child in module.named_children(): 
            res.update(find_MoE_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1, model_name = model_name
            ))
        return res

    return {}
