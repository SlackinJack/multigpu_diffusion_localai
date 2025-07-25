import argparse
import base64
import json
import logging
import os
import pickle
import safetensors
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AnimateDiffControlNetPipeline,
    AnimateDiffPipeline,
    AutoencoderKL,
    ControlNetModel,
    MotionAdapter,
    QuantoConfig,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
)
from diffusers.utils import load_image
from flask import Flask, request, jsonify
from PIL import Image

from AsyncDiff.asyncdiff.async_animate import AsyncDiff as AsyncDiffAD
from AsyncDiff.asyncdiff.async_sd import AsyncDiff as AsyncDiffSD

from modules.host_generics import *
from modules.scheduler_config import get_scheduler

app = Flask(__name__)
async_diff = None
initialized = False
local_rank = None
logger = None
pipe = None


def get_args():
    parser = argparse.ArgumentParser()
    # asyncdiff
    parser.add_argument("--model_n", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--stride", type=int, default=1, choices=[1, 2])
    parser.add_argument("--warm_up", type=int, default=3)
    parser.add_argument("--time_shift", type=bool, default=False)
    # generic
    for k,v in GENERIC_HOST_ARGS.items():   parser.add_argument(f"--{k}", type=v, default=None)
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
    global pipe, local_rank, async_diff, initialized
    args = get_args()

    # checks
    assert not (args.type == "ad" and args.motion_adapter is None), "AnimateDiff requires providing a motion adapter."

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
    if args.quantize_encoder:
        q_config = QuantoConfig(weights_dtype=args.quantize_encoder_type, activations_dtype=args.quantize_encoder_type)

    # set control net
    controlnet_model = None
    if args.control_net is not None:
        controlnet_model = ControlNetModel.from_pretrained(
            args.control_net,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
            quantization_config=q_config,
        )

    # set pipeline
    match args.type:
        case "ad":
            adapter = MotionAdapter.from_pretrained(
                args.motion_adapter,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                quantization_config=q_config,
            )
            pipe_class = AnimateDiffControlNetPipeline if args.control_net is not None else AnimateDiffPipeline
            pipe = pipe_class.from_pretrained(
                args.model,
                motion_adapter=adapter,
                controlnet=controlnet_model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                quantization_config=q_config,
            )
            if args.ip_adapter is not None:
                ip_adapter = json.loads(args.ip_adapter)
                for m,s in ip_adapter.items():
                    split = m.split("/")
                    ip_adapter_file = split[-1]
                    ip_adapter_subfolder = split[-2]
                    ip_adapter_folder = m.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "")
                    pipe.load_ip_adapter(
                        ip_adapter_folder,
                        subfolder=ip_adapter_subfolder,
                        weight_name=ip_adapter_file,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                        quantization_config=q_config,
                    )
                    pipe.set_ip_adapter_scale(s)
        case "sd1":
            pipe_class = StableDiffusionControlNetPipeline if args.control_net is not None else StableDiffusionPipeline
            pipe = pipe_class.from_pretrained(
                args.model,
                controlnet=controlnet_model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                quantization_config=q_config,
            )
        case "sd2":
            pipe_class = StableDiffusionControlNetPipeline if args.control_net is not None else StableDiffusionPipeline
            pipe = pipe_class.from_pretrained(
                args.model,
                controlnet=controlnet_model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                quantization_config=q_config,
            )
        case "sd3":
            pipe_class = StableDiffusion3ControlNetPipeline if args.control_net is not None else StableDiffusion3Pipeline
            pipe = pipeline_class.from_pretrained(
                args.model,
                controlnet=controlnet_model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                quantization_config=q_config,
            )
        case "sdup":
            pipe = StableDiffusionUpscalePipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                quantization_config=q_config,
            )
        case "sdxl":
            pipe_class = StableDiffusionXLControlNetPipeline if args.control_net is not None else StableDiffusionXLPipeline
            pipe = pipe_class.from_pretrained(
                args.model,
                controlnet=controlnet_model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                quantization_config=q_config,
            )
        case "svd":
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                quantization_config=q_config,
            )
        case _: raise NotImplementedError

    # memory saving functions
    if not args.type in ["svd"]:
        if args.enable_vae_slicing: pipe.enable_vae_slicing()
        if args.enable_vae_tiling:  pipe.enable_vae_tiling()
        if args.xformers_efficient: pipe.enable_xformers_memory_efficient_attention()

    # set asyncdiff
    if args.type in ["ad"]: ad_class = AsyncDiffAD
    else:                   ad_class = AsyncDiffSD
    async_diff = ad_class(pipe, args.type, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

    # set scheduler
    if args.type in ["ad", "sd1", "sd2", "sd3", "sdup", "sdxl"]:
        pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)
        #if args.scheduler is not None:
        #    scheduler_config = {}
        #    for k, v in pipe.scheduler.config.items():
        #        scheduler_config[k] = v
        #    pipe.scheduler = get_scheduler(args.scheduler, scheduler_config)
        #elif pipe.scheduler is None:
        #    logger.info("No scheduler provided - using DDIM")
        #    scheduler = DDIMScheduler.from_pretrained(
        #        args.model,
        #        subfolder="scheduler",
        #        clip_sample=False,
        #        timestep_spacing="linspace",
        #        beta_schedule="linear",
        #        steps_offset=1,
        #        local_files_only=True,
        #        low_cpu_mem_usage=True,
        #    )
        #    pipe.scheduler = scheduler

    # set vae
    # TODO: set vae

    # set lora
    adapter_names = None
    if args.lora is not None and args.type in ["sd1", "sd2", "sd3", "sdxl"]:
        adapter_names = load_lora(args.lora, pipe, local_rank)

    # compiles
    if args.compile_unet:           compile_unet(pipe, adapter_names)
    if args.compile_vae:            compile_vae(pipe)
    if args.compile_text_encoder:   compile_text_encoder(pipe)

    # set progress bar visibility
    pipe.set_progress_bar_config(disable=dist.get_rank() != 0)

    # warm up run
    if args.warm_up_steps > 0:
        logger.info("Starting warmup run")
        def get_warmup_image():
            # image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
            image = load_image("resources/rocket.png") # 1024x576 pixels
            image = image.resize((768, 432), Image.Resampling.LANCZOS)
            return image
        generator = torch.Generator(device="cpu").manual_seed(1)
        async_diff.reset_state(warm_up=args.warm_up)

        warm_up_frames = 25
        warm_up_cfg = 7
        warm_up_prompt = "rocket, cinematic, high dynamic range, good image contrast sharpness details"
        warm_up_negative_prompt = "wrong, disfigured, artifacts"
        warm_up_chunk_size = 8
        warm_up_width = 512
        warm_up_height = 512

        match args.type:
            case "ad":
                if args.ip_adapter is not None and args.control_net is not None:
                    pipe(
                        prompt=warm_up_prompt,
                        negative_prompt=warm_up_negative_prompt,
                        ip_adapter_image=get_warmup_image(),
                        conditioning_frames=[get_warmup_image()] * warm_up_frames,
                        num_frames=warm_up_frames,
                        guidance_scale=warm_up_cfg,
                        num_inference_steps=args.warm_up_steps,
                        generator=generator,
                    )
                elif args.ip_adapter is not None and args.control_net is None:
                    pipe(
                        prompt=warm_up_prompt,
                        negative_prompt=warm_up_negative_prompt,
                        ip_adapter_image=get_warmup_image(),
                        num_frames=warm_up_frames,
                        guidance_scale=warm_up_cfg,
                        num_inference_steps=args.warm_up_steps,
                        generator=generator,
                    )
                else:
                    pipe(
                        prompt=warm_up_prompt,
                        negative_prompt=warm_up_negative_prompt,
                        num_frames=warm_up_frames,
                        guidance_scale=warm_up_cfg,
                        num_inference_steps=args.warm_up_steps,
                        generator=generator,
                    )
            case "sdup":
                pipe(
                    prompt=warm_up_prompt,
                    negative_prompt=warm_up_negative_prompt,
                    image=get_warmup_image(),
                    num_inference_steps=args.warm_up_steps,
                    generator=generator,
                )
            case "svd":
                pipe(
                    image=get_warmup_image(),
                    decode_chunk_size=warm_up_chunk_size,
                    num_inference_steps=args.warm_up_steps,
                    width=warm_up_width,
                    height=warm_up_height,
                    generator=generator,
                )
            case _:
                pipe(
                    prompt=warm_up_prompt,
                    negative_prompt=warm_up_negative_prompt,
                    num_inference_steps=args.warm_up_steps,
                    width=warm_up_width,
                    height=warm_up_height,
                    generator=generator,
                )

    # complete
    torch.cuda.empty_cache()
    logger.info("Model initialization completed")
    initialized = True
    return


def generate_image_parallel(positive, negative, image, steps, cfg, control_net_scale, seed, frames, decode_chunk_size, clip_skip, motion_bucket_id, noise_aug_strength):
    global async_diff, pipe
    args = get_args()
    torch.cuda.reset_peak_memory_stats()
    async_diff.reset_state(warm_up=args.warm_up)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    if (args.type in ["sdup", "svd"]) or (args.type == "ad" and args.ip_adapter is not None):
        assert image is not None, "No image provided for an image pipeline."
        image = load_image(image)
        #if args.scale_input:
        #    percentage = args.scale_percentage / 100
        #    image = image.resize((int(image.size[0] * percentage), int(image.size[1] * percentage)), Image.Resampling.LANCZOS)

    match args.type:
        case "ad":
            if image is not None and args.control_net is not None and args.ip_adapter is not None:
                output = pipe(
                    prompt=positive,
                    negative_prompt=negative,
                    ip_adapter_image=image,
                    conditioning_frames=[image] * frames,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                    num_frames=frames,
                    output_type="pil",
                ).frames[0]
            elif image is not None and args.ip_adapter is not None and args.control_net is None:
                output = pipe(
                    prompt=positive,
                    negative_prompt=negative,
                    ip_adapter_image=image,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                    num_frames=frames,
                    output_type="pil",
                ).frames[0]
            else:
                output = pipe(
                    prompt=positive,
                    negative_prompt=negative,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=generator,
                    num_frames=frames,
                    output_type="pil",
                ).frames[0]
        case "sdup":
            output = pipe(
                prompt=positive,
                negative_prompt=negative,
                image=image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                output_type="pil",
            ).images[0]
        case "svd":
            output = pipe(
                image,
                width=args.width,
                height=args.height,
                num_inference_steps=steps,
                min_guidance_scale=cfg,
                generator=generator,
                num_frames=frames,
                decode_chunk_size=decode_chunk_size,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                output_type="pil",
            ).frames[0]
        case _:
            positive_embeds = None
            positive_pooled_embeds = None
            negative_embeds = None
            negative_pooled_embeds = None
            can_use_compel = args.compel and args.type in ["sd1", "sd2", "sdxl"]
            if can_use_compel:
                if args.type in ["sd1", "sd2"]: embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                else:                           embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                compel = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=embeddings_type,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )
                positive_embeds, positive_pooled_embeds = compel([positive])
                if negative and len(negative) > 0:
                    negative_embeds, negative_pooled_embeds = compel([negative])
            output = pipe(
                prompt=positive if positive_embeds is None else None,
                negative_prompt=negative if negative_embeds is None else None,
                prompt_embeds=positive_embeds,
                pooled_prompt_embeds=positive_pooled_embeds,
                negative_embeds=negative_embeds,
                negative_pooled_embeds=negative_pooled_embeds,
                width=args.width,
                height=args.height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
                clip_skip=clip_skip,
                output_type="pil",
            ).images[0]
            if can_use_compel:
                # https://github.com/damian0815/compel/issues/24
                positive_embeds = positive_pooled_embeds = negative_embeds = negative_pooled_embeds = None

    torch.cuda.empty_cache()

    if dist.get_rank() == 0:
        is_image = not args.type in ["ad", "svd"]
        if output is not None:
            pickled = pickle.dumps(output)
            output_base64 = base64.b64encode(pickled).decode('utf-8')
            return output_base64, is_image
        else:
            return None, is_image


@app.route("/generate", methods=["POST"])
def generate_image():
    args = get_args()

    logger.info("Received POST request for image generation")
    data = request.json
    positive            = data.get("positive")
    negative            = data.get("negative")
    image               = data.get("image")
    steps               = data.get("steps")
    cfg                 = data.get("cfg")
    controlnet_scale    = data.get("controlnet_scale")
    seed                = data.get("seed")
    frames              = data.get("frames")
    decode_chunk_size   = data.get("decode_chunk_size")
    clip_skip           = data.get("clip_skip")
    motion_bucket_id    = data.get("motion_bucket_id")
    noise_aug_strength  = data.get("noise_aug_strength")

    assert (image is not None or len(positive) > 0), "No input provided"
    if image is not None: image = Image.open(image)

    params = [positive, negative, image, steps, cfg, controlnet_scale, seed, frames, decode_chunk_size, clip_skip, motion_bucket_id, noise_aug_strength]
    dist.broadcast_object_list(params, src=0)
    output_base64, is_image = generate_image_parallel(*params)

    response = {
        "output": output_base64,
        "is_image": is_image,
    }

    return jsonify(response)


def run_host():
    args = get_args()
    if dist.get_rank() == 0:
        logger.info("Starting Flask host on rank 0")
        app.run(host="localhost", port=args.port)
    else:
        while True:
            params = [None] * 12 # len(params) of generate_image_parallel()
            logger.info(f"Rank {dist.get_rank()} waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None and params[2] is None:
                logger.info("Received exit signal, shutting down")
                break
            logger.info(f"Received task with parameters: {params}")
            generate_image_parallel(*params)


if __name__ == "__main__":
    initialize()
    run_host()
