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
    AnimateDiffControlNetPipeline, AnimateDiffPipeline, ControlNetModel, MotionAdapter,
    StableDiffusion3ControlNetPipeline, StableDiffusion3Pipeline, StableDiffusionControlNetPipeline,
    StableDiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline, StableVideoDiffusionPipeline,
)
from diffusers.utils import load_image
from flask import Flask, request, jsonify
from PIL import Image

from AsyncDiff.asyncdiff.async_animate import AsyncDiff as AsyncDiffAD
from AsyncDiff.asyncdiff.async_sd import AsyncDiff as AsyncDiffSD

from modules.custom_lora_loader import convert_name_to_bin, merge_weight
from modules.scheduler_config import get_scheduler

app = Flask(__name__)
async_diff = None
initialized = False
local_rank = None
logger = None
pipe = None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--stride", type=int, default=1, choices=[1, 2])
    parser.add_argument("--warm_up", type=int, default=3)
    parser.add_argument("--time_shift", type=bool, default=False)

    # added args
    parser.add_argument("--warm_up_steps", type=int, default=40)
    parser.add_argument("--port", type=int, default=6000, help="Listening port number")
    parser.add_argument("--pipeline_type", type=str, default=None, choices=["ad", "sd1", "sd2", "sd3", "sdup", "sdxl", "svd"])
    parser.add_argument("--compel", action="store_true", help="Enable Compel")
    parser.add_argument("--motion_adapter", type=str, default=None)
    parser.add_argument("--variant", type=str, default="fp16", help="PyTorch variant [fp16/fp32]")
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--lora", type=str, default=None, help="A dictionary of LoRAs to load, with their weights")
    parser.add_argument("--ip_adapter", type=str, default=None, help="IPAdapter model")
    parser.add_argument("--ip_adapter_scale", type=float, default=1.0, help="IPAdapter scale")
    parser.add_argument("--control_net", type=str, default=None, help="ControlNet model")
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true")
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--enable_slicing", action="store_true")
    parser.add_argument("--xformers_efficient", action="store_true")
    parser.add_argument("--scale_input", action="store_true")
    parser.add_argument("--scale_percentage", type=float, default=100.0)
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
    if initialized: return jsonify({"status": "initialized"}), 200
    else:           return jsonify({"status": "initializing"}), 202


def initialize():
    args = get_args()

    # asserts
    assert args.model is not None, "You need to provide a model to load."
    assert args.variant in ["fp16", "fp32"], f"Unsupported variant: {args.variant}"
    assert args.pipeline_type is not None, "No pipeline type provided."
    assert not (args.pipeline_type == "ad" and args.motion_adapter is None), "No motion adapter provided."

    # init distributed inference
    global pipe, local_rank, async_diff, initialized
    mp.set_start_method("spawn", force=True)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    setup_logger()
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    # set torch type
    match args.variant:
        case "fp16":
            torch_dtype = torch.float16
        case _:
            torch_dtype = torch.float32

    # set control net
    controlnet_model = None
    if args.control_net is not None:
        controlnet_model = ControlNetModel.from_pretrained(
            args.control_net,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )

    # set pipeline
    match args.pipeline_type:
        case "ad":
            adapter = MotionAdapter.from_pretrained(
                args.motion_adapter,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            pipe_class = AnimateDiffControlNetPipeline if args.control_net is not None else AnimateDiffPipeline
            pipe = pipe_class.from_pretrained(
                args.model,
                motion_adapter=adapter,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                controlnet=controlnet_model,
            )
            if args.ip_adapter is not None:
                split = args.ip_adapter.split("/")
                ip_adapter_file = split[-1]
                ip_adapter_subfolder = split[-2]
                ip_adapter_folder = args.ip_adapter.replace(f'/{ip_adapter_subfolder}/{ip_adapter_file}', "")
                pipe.load_ip_adapter(
                    ip_adapter_folder,
                    subfolder=ip_adapter_subfolder,
                    weight_name=ip_adapter_file,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                pipe.set_ip_adapter_scale(args.ip_adapter_scale)
        case "sd1":
            pipe_class = StableDiffusionControlNetPipeline if args.control_net is not None else StableDiffusionPipeline
            pipe = pipe_class.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                controlnet=controlnet_model,
            )
        case "sd2":
            pipe_class = StableDiffusionControlNetPipeline if args.control_net is not None else StableDiffusionPipeline
            pipe = pipe_class.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                controlnet=controlnet_model,
            )
        case "sd3":
            pipe_class = StableDiffusion3ControlNetPipeline if args.control_net is not None else StableDiffusion3Pipeline
            pipe = pipeline_class.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                controlnet=controlnet_model,
            )
        case "sdup":
            pipe = StableDiffusionUpscalePipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        case "sdxl":
            pipe_class = StableDiffusionXLControlNetPipeline if args.control_net is not None else StableDiffusionXLPipeline
            pipe = pipe_class.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                controlnet=controlnet_model,
            )
        case "svd":
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
        case _: raise NotImplementedError

    # set scheduler
    if args.pipeline_type in ["ad", "sd1", "sd2", "sd3", "sdup", "sdxl"]:
        if args.scheduler is not None:
            scheduler_config = {}
            for k, v in pipe.scheduler.config.items():
                scheduler_config[k] = v
            pipe.scheduler = get_scheduler(args.scheduler, scheduler_config)
        elif pipe.scheduler is None:
            logger.info("No scheduler provided - using DDIM")
            scheduler = DDIMScheduler.from_pretrained(
                args.model,
                subfolder="scheduler",
                clip_sample=False,
                timestep_spacing="linspace",
                beta_schedule="linear",
                steps_offset=1,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            pipe.scheduler = scheduler

    # set asyncdiff
    if args.pipeline_type in ["ad"]:    ad_class = AsyncDiffAD
    else:                               ad_class = AsyncDiffSD
    async_diff = ad_class(
        pipe,
        args.pipeline_type,
        model_n=args.model_n,
        stride=args.stride,
        time_shift=args.time_shift,
    )

    if args.lora is not None and args.pipeline_type in ["sd1", "sd2", "sd3", "sdxl"]:
        pipe.unet.enable_lora()
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
                f = torch.load(adapter, weights_only=True, map_location=torch.device(f"cuda:{local_rank}"))
                for k in f.keys():
                    merged_weights = merge_weight(local_rank, merged_weights, k, f[k], scale, len(loras))
            logger.info(f"Added LoRA[{i}], scale={scale}: {adapter}")
            i += 1

        pipe.unet.load_attn_procs(merged_weights)
        logger.info(f'Total loaded LoRAs: {i}')

    # memory saving functions
    if not args.pipeline_type in ["svd"]:
        if args.enable_slicing:
            pipe.enable_vae_slicing()
        if args.enable_tiling:
            pipe.enable_vae_tiling()
        if args.xformers_efficient:
            pipe.enable_xformers_memory_efficient_attention()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()

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
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        async_diff.reset_state(warm_up=args.warm_up)

        warm_up_frames = 25
        warm_up_cfg = 7
        warm_up_prompt = "puppy, perfect, cinematic, high dynamic range, good image contrast sharpness details"
        warm_up_negative_prompt = "wrong, disfigured, artifacts"
        warm_up_chunk_size = 8
        warm_up_width = 512
        warm_up_height = 512

        match args.pipeline_type:
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

        torch.cuda.empty_cache()

    # finish
    logger.info("Model initialization completed")
    initialized = True
    return


def generate_image_parallel(
    positive_prompt,
    negative_prompt,
    image,
    width,
    height,
    num_inference_steps,
    cfg,
    control_net_scale,
    seed,
    num_frames,
    decode_chunk_size,
    clip_skip,
    motion_bucket_id,
    noise_aug_strength
):
    global async_diff, pipe
    args = get_args()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    async_diff.reset_state(warm_up=args.warm_up)

    if (args.pipeline_type in ["sdup", "svd"]) or (args.pipeline_type == "ad" and args.ip_adapter is not None):
        assert image is not None, "No image provided for an image pipeline."
        image = load_image(image)
        if args.scale_input:
            percentage = args.scale_percentage / 100
            image = image.resize((int(image.size[0] * percentage), int(image.size[1] * percentage)), Image.Resampling.LANCZOS)

    match args.pipeline_type:
        case "ad":
            if image is not None and args.control_net is not None and args.ip_adapter is not None:
                logger.info(
                    "Active request parameters:\n"
                    f"image provided\n"
                    f"positive_prompt={positive_prompt}\n"
                    f"negative_prompt={negative_prompt}\n"
                    f"width={width}\n"
                    f"height={height}\n"
                    f"steps={num_inference_steps}\n"
                    f"cfg={cfg}\n"
                    f"seed={seed}\n"
                    f"num_frames={num_frames}\n"
                    f"control_net_scale={control_net_scale}\n"
                )
                output = pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    ip_adapter_image=image,
                    conditioning_frames=[image] * num_frames,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=cfg,
                    generator=generator,
                    num_frames=num_frames,
                    output_type="pil",
                ).frames[0]
            elif image is not None and args.ip_adapter is not None and args.control_net is None:
                logger.info(
                    "Active request parameters:\n"
                    "image provided\n"
                    f"positive_prompt={positive_prompt}\n"
                    f"negative_prompt={negative_prompt}\n"
                    f"width={width}\n"
                    f"height={height}\n"
                    f"steps={num_inference_steps}\n"
                    f"cfg={cfg}\n"
                    f"seed={seed}\n"
                    f"num_frames={num_frames}\n"
                )
                output = pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    ip_adapter_image=image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=cfg,
                    generator=generator,
                    num_frames=num_frames,
                    output_type="pil",
                ).frames[0]
            else:
                logger.info(
                    "Active request parameters:\n"
                    "no image provided\n"
                    f"positive_prompt={positive_prompt}\n"
                    f"negative_prompt={negative_prompt}\n"
                    f"width={width}\n"
                    f"height={height}\n"
                    f"steps={num_inference_steps}\n"
                    f"cfg={cfg}\n"
                    f"seed={seed}\n"
                    f"num_frames={num_frames}\n"
                )
                output = pipe(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=cfg,
                    generator=generator,
                    num_frames=num_frames,
                    output_type="pil",
                ).frames[0]
        case "sdup":
            logger.info(
                "Active request parameters:\n"
                f"image provided\n"
                f"positive_prompt={positive_prompt}\n"
                f"negative_prompt={negative_prompt}\n"
                f"steps={num_inference_steps}\n"
                f"cfg={cfg}\n"
                f"seed={seed}\n"
            )
            output = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg,
                generator=generator,
                output_type="pil",
            ).images[0]
        case "svd":
            logger.info(
                "Active request parameters:\n"
                f"image provided\n"
                f"width={width}\n"
                f"height={height}\n"
                f"steps={num_inference_steps}\n"
                f"cfg={cfg}\n"
                f"seed={seed}\n"
                f"num_frames={num_frames}\n"
                f"decode_chunk_size={decode_chunk_size}\n"
                f"motion_bucket_id={motion_bucket_id}\n"
                f"noise_aug_strength={noise_aug_strength}\n"
            )
            output = pipe(
                image,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                min_guidance_scale=cfg,
                generator=generator,
                num_frames=num_frames,
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
            can_use_compel = args.compel and args.pipeline_type in ["sd1", "sd2", "sdxl"]
            if can_use_compel:
                if args.pipeline_type in ["sd1", "sd2"]:
                    embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
                else:
                    embeddings_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
                compel = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=embeddings_type,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False,
                )
                positive_embeds, positive_pooled_embeds = compel([positive_prompt])
                if negative_prompt and len(negative_prompt) > 0:
                    negative_embeds, negative_pooled_embeds = compel([negative_prompt])
            logger.info(
                "Active request parameters:\n"
                f"positive_prompt={positive_prompt}\n"
                f"negative_prompt={negative_prompt}\n"
                f"width={width}\n"
                f"height={height}\n"
                f"steps={num_inference_steps}\n"
                f"cfg={cfg}\n"
                f"seed={seed}\n"
                f"clip_skip={clip_skip}\n"
            )
            output = pipe(
                prompt=positive_prompt if positive_embeds is None else None,
                negative_prompt=negative_prompt if negative_embeds is None else None,
                prompt_embeds=positive_embeds,
                pooled_prompt_embeds=positive_pooled_embeds,
                negative_embeds=negative_embeds,
                negative_pooled_embeds=negative_pooled_embeds,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg,
                generator=generator,
                clip_skip=clip_skip,
                output_type="pil",
            ).images[0]
            if can_use_compel:
                # https://github.com/damian0815/compel/issues/24
                positive_embeds = positive_pooled_embeds = negative_embeds = negative_pooled_embeds = None

    end_time = time.time()
    elapsed_time = end_time - start_time

    torch.cuda.empty_cache()

    if dist.get_rank() == 0:
        is_image = not args.pipeline_type in ["ad", "svd"]
        if output is not None:
            pickled = pickle.dumps(output)
            output_base64 = base64.b64encode(pickled).decode('utf-8')
            return output_base64, elapsed_time, is_image
        else:
            return None, elapsed_time, is_image


@app.route("/generate", methods=["POST"])
def generate_image():
    args = get_args()

    logger.info("Received POST request for image generation")
    data = request.json
    positive_prompt     = data.get("positive_prompt", None)
    negative_prompt     = data.get("negative_prompt", None)
    image               = data.get("image", None)
    width               = data.get("width")
    height              = data.get("height")
    num_inference_steps = data.get("num_inference_steps")
    guidance_scale      = data.get("cfg")
    controlnet_scale    = data.get("controlnet_scale")
    seed                = data.get("seed")
    num_frames          = data.get("num_frames")
    decode_chunk_size   = data.get("decode_chunk_size")
    clip_skip           = data.get("clip_skip")
    motion_bucket_id    = data.get("motion_bucket_id")
    noise_aug_strength  = data.get("noise_aug_strength")

    assert (image is not None or len(positive_prompt) > 0), "No input provided"

    if image is not None:
        image = Image.open(image)

    params = [
        positive_prompt,
        negative_prompt,
        image,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        controlnet_scale,
        seed,
        num_frames,
        decode_chunk_size,
        clip_skip,
        motion_bucket_id,
        noise_aug_strength
    ]
    dist.broadcast_object_list(params, src=0)
    output_base64, elapsed_time, is_image = generate_image_parallel(*params)

    response = {
        "message": "Image generated successfully",
        "elapsed_time": f"{elapsed_time:.2f} sec",
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
            params = [None] * 14 # len(params) of generate_image_parallel()
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
