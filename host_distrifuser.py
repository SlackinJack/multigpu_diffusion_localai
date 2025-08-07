import argparse
import base64
import logging
import os
import pickle
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from compel import Compel, ReturnedEmbeddingsType
from diffusers import QuantoConfig
from flask import Flask, request, jsonify

from DistriFuser.distrifuser.utils import DistriConfig
from DistriFuser.distrifuser.pipelines import DistriSDPipeline, DistriSDXLPipeline

from modules.host_generics import *
from modules.scheduler_config import get_scheduler

app = Flask(__name__)
initialized = False
local_rank = None
logger = None
pipe = None


def get_args():
    parser = argparse.ArgumentParser()
    # distrifuser
    parser.add_argument("--no_cuda_graph", action="store_true")
    parser.add_argument("--no_split_batch", action="store_true")
    parser.add_argument("--parallelism", type=str, default="patch", choices=["patch", "tensor", "naive_patch"])
    parser.add_argument("--split_scheme", type=str, default="row", choices=["row", "col", "alternate"])
    parser.add_argument("--sync_mode", type=str, default="corrected_async_gn", choices=[
                                                                                            "separate_gn",
                                                                                            "stale_gn",
                                                                                            "corrected_async_gn",
                                                                                            "sync_gn",
                                                                                            "full_sync",
                                                                                            "no_sync"
                                                                                        ])
    # generic
    for k, v in GENERIC_HOST_ARGS.items():  parser.add_argument(f"--{k}", type=v, default=None)
    for e in GENERIC_HOST_ARGS_TOGGLES:     parser.add_argument(f"--{e}", action="store_true")
    args = parser.parse_args()
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
    global pipe, local_rank, initialized
    args = get_args()

    # checks
    # TODO: checks

    # init distributed inference
    mp.set_start_method("spawn", force=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    setup_logger()
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    # set torch type
    torch_dtype = get_torch_type(args.variant)

    # quantize
    q_config = None
    if args.quantize_to:
        q_config = QuantoConfig(weights_dtype=args.quantize_to)

    # set distrifuser
    distri_config = DistriConfig(
        height=args.height,
        width=args.width,
        do_classifier_free_guidance=True,
        split_batch=not args.no_split_batch,
        warmup_steps=args.warm_up_steps,
        comm_checkpoint=60,
        mode=args.sync_mode,
        use_cuda_graph=not args.no_cuda_graph,
        parallelism=args.parallelism,
        split_scheme=args.split_scheme,
        verbose=True,
    )

    # set pipeline
    PipelineClass = DistriSDXLPipeline if args.type == "sdxl" else DistriSDPipeline
    pipe = PipelineClass.from_pretrained(
        pretrained_model_name_or_path=args.model,
        distri_config=distri_config,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        quantization_config=q_config,
    )

    # set memory saving
    if args.enable_vae_slicing: pipe.pipeline.enable_vae_slicing()
    if args.enable_vae_tiling:  pipe.pipeline.enable_vae_tiling()
    if args.xformers_efficient: pipe.pipeline.enable_xformers_memory_efficient_attention()

    # set ipadapter
    # TODO: set ipadapter

    # set scheduler
    pipe.pipeline.scheduler = get_scheduler(args.scheduler, pipe.pipeline.scheduler.config)

    # set vae
    # TODO: set vae

    # set lora
    adapter_names = None
    if args.lora:
        adapter_names = load_lora(args.lora, pipe.pipeline, local_rank)

    # compiles
    if args.compile_unet:           compile_unet(pipe.pipeline, adapter_names, is_distrifuser=True)
    if args.compile_vae:            compile_vae(pipe.pipeline)
    if args.compile_text_encoder:   compile_text_encoder(pipe.pipeline)

    # set progress bar visibility
    pipe.set_progress_bar_config(disable=distri_config.rank != 0)

    # warm up run
    # TODO: warm up run

    # complete
    torch.cuda.empty_cache()
    logger.info("Model initialization completed")
    initialized = True
    return


def generate_image_parallel(positive, negative, steps, seed, cfg, clip_skip):
    global pipe, local_rank
    args = get_args()
    torch.cuda.reset_peak_memory_stats()

    generator = torch.Generator(device="cpu").manual_seed(seed)

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
        positive_embeds, positive_pooled_embeds = compel([positive])
        if negative and len(negative) > 0:
            negative_embeds, negative_pooled_embeds = compel([negative])
    
    output = pipe(
        prompt=positive if positive_embeds is None else None,
        negative_prompt=negative if negative_embeds is None else None,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=cfg,
        clip_skip=clip_skip,
        prompt_embeds=positive_embeds,
        pooled_prompt_embeds=positive_pooled_embeds,
        negative_embeds=negative_embeds,
        negative_pooled_embeds=negative_pooled_embeds,
    )

    if args.compel:
        # https://github.com/damian0815/compel/issues/24
        positive_embeds = positive_pooled_embeds = negative_embeds = negative_pooled_embeds = None

    if dist.get_rank() != 0:
        # serialize output object
        output_bytes = pickle.dumps(output)

        # send output to rank 0
        dist.send(torch.tensor(len(output_bytes), device=f"cuda:{local_rank}"), dst=0)
        dist.send(torch.ByteTensor(list(output_bytes)).to(f"cuda:{local_rank}"), dst=0)

        logger.info("Output sent to rank 0")

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
            pickled_image = pickle.dumps(output)
            output_base64 = base64.b64encode(pickled_image).decode('utf-8')
            return output_base64


@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.json
    positive            = data.get("positive")
    negative            = data.get("negative")
    steps               = data.get("steps")
    seed                = data.get("seed")
    cfg                 = data.get("cfg")
    clip_skip           = data.get("clip_skip")

    assert (positive is not None and len(positive) > 0), "No input provided"

    params = [positive, negative, steps, seed, cfg, clip_skip]
    dist.broadcast_object_list(params, src=0)
    output_base64 = generate_image_parallel(*params)
    response = { "output": output_base64, "is_image": True }
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
