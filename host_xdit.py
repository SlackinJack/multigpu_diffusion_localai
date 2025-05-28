import argparse
import copy
import os
import time
import torch
import torch.distributed as dist
import pickle
import json
import logging
import base64
import torch.multiprocessing as mp
import safetensors

from compel import Compel, ReturnedEmbeddingsType
from PIL import Image
from flask import Flask, request, jsonify

from xDiT.xfuser import xFuserFluxPipeline, xFuserArgs
from xDiT.xfuser.config import FlexibleArgumentParser
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

from modules.custom_lora_loader import convert_name_to_bin, merge_weight

app = Flask(__name__)

os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

pipe = None
engine_config = None
input_config = None
local_rank = None
logger = None
initialized = False


def get_scheduler(scheduler_name, current_scheduler_config):
    scheduler_config = get_scheduler_config(scheduler_name, current_scheduler_config)
    scheduler_class = get_scheduler_class(scheduler_name)
    return scheduler_class.from_config(scheduler_config)


def get_scheduler_class(scheduler_name):
    name = scheduler_name.replace("k_", "", 1)
    match name:
        case "ddim":            return DDIMScheduler
        case "euler":           return EulerDiscreteScheduler
        case "euler_a":         return EulerAncestralDiscreteScheduler
        case "dpm_2":           return KDPM2DiscreteScheduler
        case "dpm_2_a":         return KDPM2AncestralDiscreteScheduler
        case "dpmpp_2m":        return DPMSolverMultistepScheduler
        case "dpmpp_2m_sde":    return DPMSolverMultistepScheduler
        case "dpmpp_sde":       return DPMSolverSinglestepScheduler
        case "heun":            return HeunDiscreteScheduler
        case "lms":             return LMSDiscreteScheduler
        case "pndm":            return PNDMScheduler
        case "unipc":           return UniPCMultistepScheduler
        case _:                 raise NotImplementedError


def get_scheduler_config(scheduler_name, current_scheduler_config):
    name = scheduler_name
    if name.startswith("k_"):
        current_scheduler_config["use_karras_sigmas"] = True
        name = scheduler_name.replace("k_", "", 1)
    match name:
        case "dpmpp_2m":
            current_scheduler_config["algorithm_type"] = "dpmsolver++"
            current_scheduler_config["solver_order"] = 2
        case "dpmpp_2m_sde":
            current_scheduler_config["algorithm_type"] = "sde-dpmsolver++"
    return current_scheduler_config


def get_args() -> argparse.Namespace:
    parser = FlexibleArgumentParser(description="xFuser Arguments")

    # Diffuser specific arguments
    parser.add_argument("--scheduler", type=str, default="dpmpp_2m", help="Scheduler name")

    # xDiT arguments
    """
    host_xdit.py [-h] [--model MODEL] [--download-dir DOWNLOAD_DIR]
                     [--trust-remote-code] [--warmup_steps WARMUP_STEPS]
                     [--use_parallel_vae] [--use_torch_compile] [--use_onediff]
                     [--use_teacache] [--use_fbcache] [--use_ray]
                     [--ray_world_size RAY_WORLD_SIZE]
                     [--dit_parallel_size DIT_PARALLEL_SIZE]
                     [--use_cfg_parallel]
                     [--data_parallel_degree DATA_PARALLEL_DEGREE]
                     [--ulysses_degree ULYSSES_DEGREE]
                     [--ring_degree RING_DEGREE]
                     [--pipefusion_parallel_degree PIPEFUSION_PARALLEL_DEGREE]
                     [--num_pipeline_patch NUM_PIPELINE_PATCH]
                     [--attn_layer_num_for_pp [ATTN_LAYER_NUM_FOR_PP ...]]
                     [--tensor_parallel_degree TENSOR_PARALLEL_DEGREE]
                     [--vae_parallel_size VAE_PARALLEL_SIZE]
                     [--split_scheme SPLIT_SCHEME] [--height HEIGHT]
                     [--width WIDTH] [--num_frames NUM_FRAMES]
                     [--img_file_path IMG_FILE_PATH] [--prompt [PROMPT ...]]
                     [--no_use_resolution_binning]
                     [--negative_prompt [NEGATIVE_PROMPT ...]]
                     [--num_inference_steps NUM_INFERENCE_STEPS]
                     [--max_sequence_length MAX_SEQUENCE_LENGTH] [--seed SEED]
                     [--output_type OUTPUT_TYPE]
                     [--guidance_scale GUIDANCE_SCALE]
                     [--enable_sequential_cpu_offload]
                     [--enable_model_cpu_offload] [--enable_tiling]
                     [--enable_slicing] [--use_fp8_t5_encoder]
                     [--use_fast_attn] [--n_calib N_CALIB]
                     [--threshold THRESHOLD] [--window_size WINDOW_SIZE]
                     [--coco_path COCO_PATH] [--use_cache]
    """

    # Added arguments
    parser.add_argument("--warm_up_steps", type=int, default=40)
    parser.add_argument("--port", type=int, default=6000, help="Listening port number")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model folder")
    #parser.add_argument("--height", type=int, default=512, help="Image height")
    #parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--variant", type=str, default="fp16", help="PyTorch variant [fp16/fp32]")
    parser.add_argument("--pipeline_type", type=str, default="flux", help="Stable Diffusion pipeline type [flux]")
    parser.add_argument("--compel", action="store_true", help="Enable Compel")
    parser.add_argument("--lora", type=str, default=None, help="A dictionary of LoRAs to load, with their weights")
    #parser.add_argument("--enable_model_cpu_offload", action="store_true")
    #parser.add_argument("--enable_sequential_cpu_offload", action="store_true")
    #parser.add_argument("--enable_tiling", action="store_true")
    #parser.add_argument("--enable_slicing", action="store_true")
    parser.add_argument("--xformers_efficient", action="store_true")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    return args


def setup_logger():
    global logger
    global local_rank
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)


@app.route("/initialize", methods=["GET"])
def check_initialize():
    global initialized
    if initialized:
        return jsonify({"status": "initialized"}), 200
    else:
        return jsonify({"status": "initializing"}), 202


def initialize():
    global pipe, engine_config, input_config, local_rank, initialized
    mp.set_start_method("spawn", force=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    setup_logger()
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    args = get_args()
    assert (args.height > 0 and args.width > 0), "Invalid image dimensions"
    assert args.model_path is not None, "No model specified"
    assert args.variant in ["fp16", "fp32"], "Unsupported variant"
    match args.variant:
        case "fp16":
            torch_dtype = torch.float16
        case _:
            torch_dtype = torch.float32

    # remove all our args before passing it to xdit
    xfuser_args = copy.deepcopy(args)
    del xfuser_args.warm_up_steps
    del xfuser_args.port
    xfuser_args.model = xfuser_args.model_path
    del xfuser_args.model_path
    # del xfuser_args.height
    # del xfuser_args.width
    del xfuser_args.variant
    del xfuser_args.pipeline_type
    del xfuser_args.compel
    del xfuser_args.lora
    #del xfuser_args.enable_model_cpu_offload
    #del xfuser_args.enable_sequential_cpu_offload
    #del xfuser_args.enable_tiling
    #del xfuser_args.enable_slicing
    del xfuser_args.xformers_efficient

    engine_args = xFuserArgs.from_cli_args(xfuser_args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch_dtype

    cache_args = {
        "use_teacache": True,
        "use_fbcache": True,
        "rel_l1_thresh": 0.12,
        "return_hidden_states_first": False,
        "num_steps": 30,
    }

    assert args.pipeline_type in ["flux"], "Unsupported pipeline"

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        engine_config=engine_config,
        cache_args=cache_args,
        torch_dtype=torch_dtype,
    )

    pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

    if args.lora:
        pipe.unet.model.enable_lora()
        loras = json.loads(args.lora)
        merged_weights = {}
        i = 0

        for adapter, scale in loras.items():
            if adapter.endswith(".safetensors"):
                safe_dict = safetensors.torch.load_file(adapter, device=f'cuda:{local_rank}')
                for k in safe_dict:
                    if ('text' in k) or ('unet' not in k) or ('transformer_blocks' not in k) or ('ff_net' in k) or ('alpha' in k):
                        continue
                    merged_weights = merge_weight(local_rank, merged_weights, convert_name_to_bin(k), safe_dict[k], scale, len(loras))
            else:
                f = torch.load(adapter, weights_only=True, map_location=torch.device(f'cuda:{local_rank}'))
                for k in f.keys():
                    merged_weights = merge_weight(local_rank, merged_weights, k, f[k], scale, len(loras))
            logger.info(f"Added LoRA[{i}], scale={scale}: {adapter}")
            i += 1

        pipe.unet.model.load_attn_procs(merged_weights)
        logger.info(f'Total loaded LoRAs: {i}')

    if args.enable_slicing:
        pipe.enable_vae_slicing()
    if args.enable_tiling:
        pipe.enable_vae_tiling()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
        #pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
    if args.xformers_efficient:
        pipe.enable_xformers_memory_efficient_attention()

    pipe = pipe.to(f"cuda:{local_rank}")

    # warmup run
    start_time = time.time()
    output = pipe(
        height=512,
        width=512,
        prompt="a dog",
        num_inference_steps=30,
        output_type="pil",
        max_sequence_length=256,
        guidance_scale=7,
        generator=torch.Generator(device="cpu").manual_seed(1),
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info("Model initialization completed")
    initialized = True
    return

"""
def generate_image_parallel(
    positive_prompt,
    negative_prompt,
    num_inference_steps,
    seed,
    cfg,
    clip_skip
):
    global pipe, local_rank, input_config
    logger.info(
        "Active request parameters:\n"
        f"positive_prompt={positive_prompt}\n"
        f"negative_prompt={negative_prompt}\n"
        f"steps={num_inference_steps}\n"
        f"seed={seed}\n"
        f"cfg={cfg}\n"
        f"clip_skip={clip_skip}\n"
    )
    logger.info(f"Starting image generation with prompt: {positive_prompt}")
    logger.info(f"Negative: {negative_prompt}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    args = get_args()

    positive_embeds = None
    positive_pooled_embeds = None
    negative_embeds = None
    negative_pooled_embeds = None
    if args.compel:
        compel = Compel(
            tokenizer=[pipe.pipeline.tokenizer, pipe.pipeline.tokenizer_2],
            text_encoder=[pipe.pipeline.text_encoder, pipe.pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False
        )
        positive_embeds, positive_pooled_embeds = compel([positive_prompt])
        if negative_prompt and len(negative_prompt) > 0:
            negative_embeds, negative_pooled_embeds = compel([negative_prompt])
    
    output = pipe(
        prompt=positive_prompt if positive_embeds is None else None,
        negative_prompt=negative_prompt if negative_embeds is None else None,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=cfg,
        clip_skip=clip_skip,
        prompt_embeds=positive_embeds,
        pooled_prompt_embeds=positive_pooled_embeds,
        negative_embeds=negative_embeds,
        negative_pooled_embeds=negative_pooled_embeds,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")

    if args.compel:
        # https://github.com/damian0815/compel/issues/24
        positive_embeds = positive_pooled_embeds = negative_embeds = negative_pooled_embeds = None

    if dist.get_rank() != 0:
        # serialize output object
        output_bytes = pickle.dumps(output)

        # send output to rank 0
        dist.send(torch.tensor(len(output_bytes), device=f"cuda:{local_rank}"), dst=0)
        dist.send(torch.ByteTensor(list(output_bytes)).to(f"cuda:{local_rank}"), dst=0)

        logger.info(f"Output sent to rank 0")

    elif dist.get_rank() == 0 and dist.get_world_size() > 1:
        # recv from rank world_size - 1
        size = torch.tensor(0, device=f"cuda:{local_rank}")
        dist.recv(size, src=dist.get_world_size() - 1)
        output_bytes = torch.ByteTensor(size.item()).to(f"cuda:{local_rank}")
        dist.recv(output_bytes, src=dist.get_world_size() - 1)

        # deserialize output object
        output = pickle.loads(output_bytes.cpu().numpy().tobytes())
        if output is not None:
            output = output.images[0]

    return output, elapsed_time, True
"""

"""
@app.route("/generate", methods=["POST"])
def generate_image():
    logger.info("Received POST request for image generation")
    data = request.json
    positive_prompt     = data.get("positive_prompt", None)
    negative_prompt     = data.get("negative_prompt", None)
    num_inference_steps = data.get("num_inference_steps")
    seed                = data.get("seed")
    cfg                 = data.get("cfg",)
    clip_skip           = data.get("clip_skip")

    logger.info(
        "Request parameters:\n"
        f"positive_prompt='{positive_prompt}'\n"
        f"negative_prompt='{negative_prompt}'\n"
        f"steps={num_inference_steps}\n"
        f"seed={seed}\n"
        f"cfg={cfg}\n"
        f"clip_skip={clip_skip}"
    )

    # Broadcast request parameters to all processes
    params = [positive_prompt, negative_prompt, num_inference_steps, seed, cfg, clip_skip]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    output, elapsed_time, is_image = generate_image_parallel(*params)
    pickled_image = pickle.dumps(output)
    output_base64 = base64.b64encode(pickled_image).decode('utf-8')

    response = {
        "message": "Image generated successfully",
        "elapsed_time": f"{elapsed_time:.2f} sec",
        "output": output_base64,
        "is_image": is_image,
    }

    logger.info("Sending response")
    return jsonify(response)
"""


def run_host():
    args = get_args()
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="localhost", port=args.port)
    #else:
    #    while True:
    #        params = [None] * 6 # len(params) of generate_image_parallel()
    #        logger.info(f"Rank {dist.get_rank()} waiting for tasks")
    #        dist.broadcast_object_list(params, src=0)
    #        if params[0] is None:
    #            logger.info("Received exit signal, shutting down")
    #            break
    #        logger.info(f"Received task with parameters: {params}")
    #        generate_image_parallel(*params)


if __name__ == "__main__":
    initialize()
    logger.info(f"Process initialized. Rank: {dist.get_rank()}, Local Rank: {os.environ.get('LOCAL_RANK', 'Not Set')}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    run_host()
