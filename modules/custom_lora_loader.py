import torch

# https://github.com/haofanwang/Lora-for-Diffusers/blob/18adfa4da0afec46679eb567d5a3690fd6a4ce9c/format_convert.py
def convert_name_to_bin(name):
    new_name = name.replace('lora_unet' + '_', '')
    new_name = new_name.replace('.weight', '')
    parts = new_name.split('.')

    #parts[0] = parts[0].replace('_0', '')
    if 'out' in parts[0]:
        parts[0] = "_".join(parts[0].split('_')[:-1])

    parts[1] = parts[1].replace('_', '.')
    sub_parts = parts[0].split('_')

    new_sub_parts = ""
    for i in range(len(sub_parts)):
        if sub_parts[i] in ['block', 'blocks', 'attentions'] or sub_parts[i].isnumeric() or 'attn' in sub_parts[i]:
            if 'attn' in sub_parts[i]:
                new_sub_parts += sub_parts[i] + ".processor."
            else:
                new_sub_parts += sub_parts[i] + "."
        else:
            new_sub_parts += sub_parts[i] + "_"

    new_sub_parts += parts[1]
    new_name =  new_sub_parts + '.weight'
    return new_name


def merge_weight(local_rank, dictionary, key, weight, scale, total_length):
    out = dictionary
    scaled_weight = torch.mul(weight, scale) #(scale / total_length))
    existing_weight = out.get(key)
    if existing_weight is None:
        out[key] = scaled_weight
    else:
        ex = list(existing_weight.size())
        sc = list(scaled_weight.size())
        x1, y1 = ex[0], ex[1]
        x2, y2 = sc[0], sc[1]
        if x2 > x1 or y2 > y1:
            new_existing_weight = torch.zeros((x2, y2), device=f'cuda:{local_rank}')
            new_existing_weight[:x1, :y1] = existing_weight
            del existing_weight
            torch.cuda.empty_cache()
            existing_weight = new_existing_weight
        elif x1 > x2 or y1 > y2:
            new_scaled_weight = torch.zeros((x1, y1), device=f'cuda:{local_rank}')
            new_scaled_weight[:x2, :y2] = scaled_weight
            del scaled_weight
            torch.cuda.empty_cache()
            scaled_weight = new_scaled_weight
        out[key] = torch.add(scaled_weight, existing_weight)
    return out
