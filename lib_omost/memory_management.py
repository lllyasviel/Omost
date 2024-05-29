# A super simple version of model management
# wHy sO seRiOuS?

import torch


high_vram = False
gpu = torch.device('cuda')
cpu = torch.device('cpu')

models_in_gpu = []


def load_models_to_gpu(models):
    global models_in_gpu

    if not isinstance(models, (tuple, list)):
        models = [models]

    models_to_load = [m for m in set(models) if m not in models_in_gpu]

    if not high_vram:
        for m in models_in_gpu:
            m.to(cpu)
            print('Unload to CPU:', m.__class__.__name__)
        models_in_gpu = []

    for m in models_to_load:
        m.to(gpu)
        print('Load to GPU:', m.__class__.__name__)

    models_in_gpu = list(set(models_in_gpu + models))
    torch.cuda.empty_cache()
    return
