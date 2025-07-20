import argparse
import base64
import copy
import io
import json
import logging
import os
import pickle
import requests
import safetensors
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import AutoencoderKL, FluxTransformer2DModel, GGUFQuantizationConfig, QuantoConfig
from flask import Flask, request, jsonify
from optimum.quanto import freeze, quantize
from PIL import Image
from transformers import T5EncoderModel

from xDiT.xfuser import xFuserFluxPipeline, xFuserStableDiffusion3Pipeline, xFuserArgs
from xDiT.xfuser.config import FlexibleArgumentParser

from modules.host_generics import *
from modules.scheduler_config import get_scheduler

app = Flask(__name__)
engine_config = None
initialized = False
input_config = None
local_rank = None
logger = None
pipe = None
result = None
cache_args = {
    "use_teacache": True,
    "use_fbcache": True,
    "rel_l1_thresh": 0.12,
    "return_hidden_states_first": False,
    "num_steps": 30,
}


def get_args():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    # xDiT arguments
    """
        [--model MODEL] [--download-dir DOWNLOAD_DIR]
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
    #generic
    for k,v in GENERIC_HOST_ARGS.items():
        if k not in ["height", "width", "model"]:
            parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:     parser.add_argument(f"--{e}", action="store_true")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    return args


def setup_logger():
    global local_rank, logger
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.INFO, format=f"[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)


@app.route("/initialize", methods=["GET"])
def check_initialize():
    global initialized
    if initialized: return "", 200
    else:           return "", 202


def initialize():
    global pipe, engine_config, input_config, local_rank, initialized, cache_args
    args = get_args()

    # checks
    # TODO: checks

    # set torch type
    torch_dtype = get_torch_type(args.variant)

    # init distributed inference
    # remove all our args before passing it to xdit
    xfuser_args = copy.deepcopy(args)
    del xfuser_args.gguf_model
    del xfuser_args.scheduler
    del xfuser_args.warm_up_steps
    del xfuser_args.port
    del xfuser_args.variant
    del xfuser_args.type
    del xfuser_args.lora
    del xfuser_args.compile_unet
    del xfuser_args.compile_vae
    del xfuser_args.compile_text_encoder
    del xfuser_args.quantize_encoder
    del xfuser_args.quantize_encoder_type
    engine_args = xFuserArgs.from_cli_args(xfuser_args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch_dtype
    local_rank = int(os.environ.get("LOCAL_RANK"))
    setup_logger()
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    # quantize
    q_config = None
    if args.quantize_encoder:
        q_config = QuantoConfig(weights_dtype=args.quantize_encoder_type, activations_dtype=args.quantize_encoder_type)

    def do_quantization(model, desc):
        logging.info(f"rank {local_rank} quantizing {desc} to {args.quantize_encoder_type}")
        weights = get_encoder_type(args.quantize_encoder_type)
        quantize(model, weights=weights)
        freeze(model)

    # set pipeline
    match args.type:
        case "flux":
            using_gguf = False
            if args.gguf_model:
                using_gguf = True
                transformer = FluxTransformer2DModel.from_single_file(
                    args.gguf_model,
                    torch_dtype=torch_dtype,
                    config=args.model+"/transformer",
                    #use_safetensors=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                )
            if using_gguf:
                if args.quantize_encoder:
                    text_encoder_2 = T5EncoderModel.from_pretrained(args.model, subfolder="text_encoder_2", torch_dtype=torch_dtype)
                    do_quantization(text_encoder_2, "text_encoder_2")
                    pipe = xFuserFluxPipeline.from_pretrained(
                        pretrained_model_name_or_path=args.model,
                        engine_config=engine_config,
                        cache_args=cache_args,
                        transformer=transformer,
                        text_encoder_2=text_encoder_2,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        quantization_config=q_config,
                    )
                else:
                    pipe = xFuserFluxPipeline.from_pretrained(
                        pretrained_model_name_or_path=args.model,
                        engine_config=engine_config,
                        cache_args=cache_args,
                        transformer=transformer,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        quantization_config=q_config,
                    )
            else:
                if args.quantize_encoder:
                    text_encoder_2 = T5EncoderModel.from_pretrained(args.model, subfolder="text_encoder_2", torch_dtype=torch_dtype)
                    do_quantization(text_encoder_2, "text_encoder_2")
                    pipe = xFuserFluxPipeline.from_pretrained(
                        pretrained_model_name_or_path=args.model,
                        engine_config=engine_config,
                        cache_args=cache_args,
                        text_encoder_2=text_encoder_2,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        quantization_config=q_config,
                    )
                else:
                    pipe = xFuserFluxPipeline.from_pretrained(
                        pretrained_model_name_or_path=args.model,
                        engine_config=engine_config,
                        cache_args=cache_args,
                        torch_dtype=torch_dtype,
                        use_safetensors=True,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        quantization_config=q_config,
                    )
        case "sd3":
            if args.quantize_encoder:
                text_encoder_3 = T5EncoderModel.from_pretrained(args.model, subfolder="text_encoder_3", torch_dtype=torch_dtype)
                do_quantization(text_encoder_3, "text_encoder_3")
                pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
                    pretrained_model_name_or_path=args.model,
                    engine_config=engine_config,
                    text_encoder_3=text_encoder_3,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    quantization_config=q_config,
                )
            else:
                pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
                    pretrained_model_name_or_path=args.model,
                    engine_config=engine_config,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    quantization_config=q_config,
                )
        case _: raise NotImplementedError

    # set memory saving
    if args.type not in ["sd3"]:
        if args.enable_vae_slicing: pipe.enable_vae_slicing()
        if args.enable_vae_tiling:  pipe.enable_vae_tiling()
    if args.xformers_efficient: pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to(f"cuda:{local_rank}")

    # set ipadapter
    # TODO: set ipadapter

    # set scheduler
    # TODO: set scheduler
    # pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

    # set vae
    # TODO: set vae

    # set lora
    adapter_names = None
    if args.lora:
        adapter_names = load_lora(args.lora, pipe, local_rank)

    # compiles
    if args.compile_unet and args.type not in ["flux", "sd3"]:  compile_unet(pipe, adapter_names)
    if args.compile_vae:                                        compile_vae(pipe)
    if args.compile_text_encoder:                               compile_text_encoder(pipe)

    # set progress bar visibility
    pipe.set_progress_bar_config(disable=local_rank != 0)

    # warm up run
    output = pipe(
        height=512,
        width=512,
        prompt="a dog",
        num_inference_steps=args.warm_up_steps,
        guidance_scale=7,
        generator=torch.Generator(device="cpu").manual_seed(1),
        max_sequence_length=256,
        output_type="pil",
        use_resolution_binning=input_config.use_resolution_binning,
    )

    # complete
    torch.cuda.empty_cache()
    logger.info("Model initialization completed")
    initialized = True
    return


def generate_image_parallel(positive, negative, steps, cfg, seed, clip_skip):
    global pipe, local_rank, input_config, result
    args = get_args()
    torch.cuda.reset_peak_memory_stats()

    generator = torch.Generator(device="cpu").manual_seed(seed)

    match args.type:
        case _:
            is_image = True
            output = pipe(
                prompt=positive,
                negative_prompt=negative,
                width=args.width,
                height=args.height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                clip_skip=clip_skip,
                max_sequence_length=256,
                output_type="pil",
                use_resolution_binning=input_config.use_resolution_binning,
            )

    torch.cuda.empty_cache()

    if local_rank == 0:
        while True:
            if result is not None:
                output_base64 = result
                result = None
                return output_base64, is_image
    elif output is not None:
        logger.info(f"Rank {local_rank} task completed")
        if is_image:    pickled = pickle.dumps(output.images[0])
        else:           pickled = pickle.dumps(output.frames[0])
        output_base64 = base64.b64encode(pickled).decode("utf-8")
        with app.app_context():
            requests.post(f"http://localhost:{args.port}/set_result", json={ "output": output_base64 })


@app.route("/set_result", methods=["POST"])
def set_result():
    global result
    data = request.json
    result = data.get("output")
    return "", 200


@app.route("/generate", methods=["POST"])
def generate_image():
    logger.info("Received POST request for image generation")
    data = request.json
    positive            = data.get("positive")
    negative            = data.get("negative")
    steps               = data.get("steps")
    cfg                 = data.get("cfg",)
    seed                = data.get("seed")
    clip_skip           = data.get("clip_skip")

    params = [
        positive,
        negative,
        steps,
        cfg,
        seed,
        clip_skip
    ]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    output_base64, is_image = generate_image_parallel(*params)

    response = {
        "output": output_base64,
        "is_image": is_image,
    }

    logger.info("Sending response")
    return jsonify(response)


def run_host():
    args = get_args()
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="localhost", port=args.port)
    else:
        while True:
            params = [None] * 6 # len(params) of generate_image_parallel()
            logger.info(f"Rank {dist.get_rank()} waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                logger.info("Received exit signal, shutting down")
                break
            logger.info(f"Received task with parameters: {params}")
            generate_image_parallel(*params)


if __name__ == "__main__":
    initialize()
    run_host()
