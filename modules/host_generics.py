import json
import safetensors
import torch
from optimum.quanto import qint2, qint4, qint8, qfloat8


config              =   json.load(open("config.json"))
config_compiler     =   config["compiler"]


GENERIC_HOST_ARGS = {
    "height":                   int,
    "width":                    int,
    "warm_up_steps":            int,
    "port":                     int,
    "type":                     str,
    "variant":                  str,
    "scheduler":                str,
    "quantize_to":              str,
    "model":                    str,    # path
    "gguf_model":               str,    # path
    "motion_adapter":           str,    # path
    "control_net":              str,    # path
    "lora":                     str,    # json dict > { "path": scale, ... }
    "ip_adapter":               str,    # json dict > { "path": scale, ... }
    "image_scale":              float,
}


GENERIC_HOST_ARGS_TOGGLES = [
    "compel",
    "enable_vae_tiling",
    "enable_vae_slicing",
    "xformers_efficient",
    "compile_unet",
    "compile_vae",
    "compile_text_encoder",
]


def get_torch_type(t):
    match t:
        case "fp16":    return torch.float16
        case "bf16":    return torch.bfloat16
        case _:         return torch.float32


def get_encoder_type(t):
    match t:
        case "int2":    return qint2
        case "int4":    return qint4
        case "int8":    return qint8
        case _:         return qfloat8


def load_lora(lora_dict, pipe, local_rank):
    loras = json.loads(lora_dict)
    names = []
    i = 0

    for m,s in loras.items():
        if m.endswith(".safetensors"):  weights = safetensors.torch.load_file(m, device=f'cuda:{local_rank}')
        else:                           weights = torch.load(m, map_location=torch.device(f'cuda:{local_rank}'))
        w = m.split("/")[-1]
        a = w if not "." in w else w.split(".")[0]
        names.append(a)
        pipe.load_lora_weights(weights, weight_name=w, adapter_name=a)
        #logger.info(f"Added LoRA[{i}], scale={s}: {m}")
        i += 1

    pipe.unet.set_adapters(names, list(loras.values()))
    #pipe.text_encoder.enable_adapters()
    #logger.info(f'Total loaded LoRAs: {i}')
    #logger.info(f'UNet Adapters: {str(pipe.unet.active_adapters())}')
    #logger.info(f'TextEncoder Adapters: {str(pipe.text_encoder.active_adapters())}')
    return names


def compile_unet(pipe, adapter_names, is_distrifuser=False):
    global config_compiler
    backend         = config_compiler["backend"]
    mode            = config_compiler["mode"]
    fullgraph       = config_compiler["fullgraph"]
    if adapter_names:
        if is_distrifuser:  pipe.unet.model.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
        else:               pipe.unet.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
        pipe.unload_lora_weights()
    if is_distrifuser:
        if len(mode) > 0:   pipe.unet.model = torch.compile(pipe.unet.model, backend=backend, mode=mode, fullgraph=fullgraph)
        else:               pipe.unet.model = torch.compile(pipe.unet.model, backend=backend, fullgraph=fullgraph)
    else:
        if len(mode) > 0:   pipe.unet = torch.compile(pipe.unet, backend=backend, mode=mode, fullgraph=fullgraph)
        else:               pipe.unet = torch.compile(pipe.unet, backend=backend, fullgraph=fullgraph)
    return


def compile_vae(pipe):
    global config_compiler
    backend         = config_compiler["backend"]
    mode            = config_compiler["mode"]
    fullgraph       = config_compiler["fullgraph"]
    if len(mode) > 0:           pipe.vae = torch.compile(pipe.vae, backend=backend, mode=mode, fullgraph=fullgraph)
    else:                       pipe.vae = torch.compile(pipe.vae, backend=backend, fullgraph=fullgraph)
    return


def compile_text_encoder(pipe):
    global config_compiler
    backend         = config_compiler["backend"]
    mode            = config_compiler["mode"]
    fullgraph       = config_compiler["fullgraph"]
    if len(mode) > 0:           pipe.text_encoder = torch.compile(pipe.text_encoder, backend=backend, mode=mode, fullgraph=fullgraph)
    else:                       pipe.text_encoder = torch.compile(pipe.text_encoder, backend=backend, fullgraph=fullgraph)
    return
