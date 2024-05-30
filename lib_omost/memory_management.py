import torch
from contextlib import contextmanager


high_vram = False
gpu = torch.device('cuda')
cpu = torch.device('cpu')

torch.zeros((1, 1)).to(gpu, torch.float32)
torch.cuda.empty_cache()

models_in_gpu = []


@contextmanager
def movable_bnb_model(m):
    if hasattr(m, 'quantization_method'):
        m.quantization_method_backup = m.quantization_method
        del m.quantization_method
    try:
        yield None
    finally:
        if hasattr(m, 'quantization_method_backup'):
            m.quantization_method = m.quantization_method_backup
            del m.quantization_method_backup
    return


def load_models_to_gpu(models):
    global models_in_gpu

    if not isinstance(models, (tuple, list)):
        models = [models]

    models_to_remain = [m for m in set(models) if m in models_in_gpu]
    models_to_load = [m for m in set(models) if m not in models_in_gpu]
    models_to_unload = [m for m in set(models_in_gpu) if m not in models_to_remain]

    if not high_vram:
        for m in models_to_unload:
            with movable_bnb_model(m):
                m.to(cpu)
            print('Unload to CPU:', m.__class__.__name__)
        models_in_gpu = models_to_remain

    for m in models_to_load:
        with movable_bnb_model(m):
            m.to(gpu)
        print('Load to GPU:', m.__class__.__name__)

    models_in_gpu = list(set(models_in_gpu + models))
    torch.cuda.empty_cache()
    return


def unload_all_models(extra_models=None):
    global models_in_gpu

    if extra_models is None:
        extra_models = []

    if not isinstance(extra_models, (tuple, list)):
        extra_models = [extra_models]

    models_in_gpu = list(set(models_in_gpu + extra_models))

    return load_models_to_gpu([])
