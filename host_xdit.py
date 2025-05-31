import argparse
import base64
import copy
import io
import json
import logging
import os
import pickle
import safetensors
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
from flask import Flask, request, jsonify
from optimum.quanto import freeze, qint2, qint4, qint8, qfloat8, quantize
from PIL import Image
from transformers import T5EncoderModel

from xDiT.xfuser import xFuserFluxPipeline, xFuserStableDiffusion3Pipeline, xFuserArgs
from xDiT.xfuser.config import FlexibleArgumentParser

from modules.custom_lora_loader import convert_name_to_bin, merge_weight
from modules.scheduler_config import get_scheduler

app = Flask(__name__)
engine_config = None
initialized = False
input_config = None
local_rank = None
logger = None
pipe = None


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

    # Added arguments
    parser.add_argument("--scheduler", type=str, default="dpmpp_2m", help="Scheduler name")
    parser.add_argument("--warm_up_steps", type=int, default=40)
    parser.add_argument("--port", type=int, default=6000, help="Listening port number")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model folder")
    parser.add_argument("--variant", type=str, default="fp16", help="PyTorch variant [fp16/fp32]")
    parser.add_argument("--pipeline_type", type=str, default=None, choices=["flux", "sd3"])
    parser.add_argument("--lora", type=str, default=None, help="A dictionary of LoRAs to load, with their weights")
    parser.add_argument("--xformers_efficient", action="store_true")
    parser.add_argument("--quantize_encoder", action="store_true")
    parser.add_argument("--quantize_encoder_type", default="qfloat8", choices=["qint2", "qint4", "qint8", "qfloat8"])
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
    if initialized: return jsonify({"status": "initialized"}), 200
    else:           return jsonify({"status": "initializing"}), 202


def initialize():
    global pipe, engine_config, input_config, local_rank, initialized

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
    del xfuser_args.scheduler
    del xfuser_args.warm_up_steps
    del xfuser_args.port
    xfuser_args.model = xfuser_args.model_path
    del xfuser_args.model_path
    del xfuser_args.variant
    del xfuser_args.pipeline_type
    del xfuser_args.lora
    del xfuser_args.quantize_encoder
    del xfuser_args.quantize_encoder_type

    engine_args = xFuserArgs.from_cli_args(xfuser_args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch_dtype

    local_rank = int(os.environ.get("LOCAL_RANK"))
    setup_logger()

    def do_quantization(model, desc):
        logging.info(f"rank {local_rank} quantizing {desc} to {args.quantize_encoder_type}")
        match args.quantize_encoder_type:
            case "qint2":
                weights = qint2
            case "qint4":
                weights = qint4
            case "qint8":
                weights = qint8
            case _:
                weights = qfloat8
        quantize(model, weights=weights)
        freeze(model)

    match args.pipeline_type:
        case "flux":
            using_gguf = False
            # GGUF loader
            if xfuser_args.model.endswith(".gguf"):
                # we expect the Flux model folder in the same location, eg:
                # models |
                #        | - FLUX.1-schnell-Q5_K_S.gguf <- Quantized GGUF transformer
                #        | - FLUX.1-schnell/            <- Base FLUX model folder
                #        | - other models
                #        | - ...
                using_gguf = True
                flux_gguf = xfuser_args.model
                parts = xfuser_args.model.split("/")
                gguf_file_name = parts[len(parts) - 1]
                folder = xfuser_args.model.replace(gguf_file_name, "")
                if "schnell" in xfuser_args.model:
                    # FLUX.1-schnell
                    flux_folder = folder + "FLUX.1-schnell"
                elif "dev" in xfuser_args.model:
                    # FLUX.1-dev
                    flux_folder = folder + "FLUX.1-dev"
                else:
                    assert False, "Unknown FLUX type"

                logging.info("Loading Flux GGUF: " + flux_gguf + ", Flux Folder: " + flux_folder)
                transformer = FluxTransformer2DModel.from_single_file(
                    flux_gguf,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                    torch_dtype=torch_dtype,
                    config=flux_folder+"/transformer",
                    local_files_only=True,
                )

            if using_gguf:
                if args.quantize_encoder:
                    text_encoder_2 = T5EncoderModel.from_pretrained(flux_folder, subfolder="text_encoder_2", torch_dtype=torch_dtype)
                    do_quantization(text_encoder_2, "text_encoder_2")
                    pipe = xFuserFluxPipeline.from_pretrained(
                        pretrained_model_name_or_path=flux_folder,
                        engine_config=engine_config,
                        cache_args={
                            "use_teacache": True,
                            "use_fbcache": True,
                            "rel_l1_thresh": 0.12,
                            "return_hidden_states_first": False,
                            "num_steps": 30,
                        },
                        transformer=transformer,
                        text_encoder_2=text_encoder_2,
                        torch_dtype=torch_dtype,
                        local_files_only=True,
                    )
                else:
                    pipe = xFuserFluxPipeline.from_pretrained(
                        pretrained_model_name_or_path=flux_folder,
                        engine_config=engine_config,
                        cache_args={
                            "use_teacache": True,
                            "use_fbcache": True,
                            "rel_l1_thresh": 0.12,
                            "return_hidden_states_first": False,
                            "num_steps": 30,
                        },
                        transformer=transformer,
                        torch_dtype=torch_dtype,
                        local_files_only=True,
                    )
            else:
                if args.quantize_encoder:
                    text_encoder_2 = T5EncoderModel.from_pretrained(xfuser_args.model, subfolder="text_encoder_2", torch_dtype=torch_dtype)
                    do_quantization(text_encoder_2, "text_encoder_2")
                    pipe = xFuserFluxPipeline.from_pretrained(
                        pretrained_model_name_or_path=xfuser_args.model,
                        engine_config=engine_config,
                        cache_args={
                            "use_teacache": True,
                            "use_fbcache": True,
                            "rel_l1_thresh": 0.12,
                            "return_hidden_states_first": False,
                            "num_steps": 30,
                        },
                        text_encoder_2=text_encoder_2,
                        torch_dtype=torch_dtype,
                        local_files_only=True,
                    )
                else:
                    pipe = xFuserFluxPipeline.from_pretrained(
                        pretrained_model_name_or_path=xfuser_args.model,
                        engine_config=engine_config,
                        cache_args={
                            "use_teacache": True,
                            "use_fbcache": True,
                            "rel_l1_thresh": 0.12,
                            "return_hidden_states_first": False,
                            "num_steps": 30,
                        },
                        torch_dtype=torch_dtype,
                        local_files_only=True,
                    )
        case "sd3":
            if args.quantize_encoder:
                text_encoder_3 = T5EncoderModel.from_pretrained(xfuser_args.model, subfolder="text_encoder_3", torch_dtype=torch_dtype)
                do_quantization(text_encoder_3, "text_encoder_3")

                pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
                    pretrained_model_name_or_path=xfuser_args.model,
                    engine_config=engine_config,
                    text_encoder_3=text_encoder_3,
                    torch_dtype=torch_dtype,
                )
            else:
                pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
                    pretrained_model_name_or_path=xfuser_args.model,
                    engine_config=engine_config,
                    torch_dtype=torch_dtype,
                )
        case _: raise NotImplementedError

    # TODO: implement scheduler
    #pipe.scheduler = get_scheduler(args.scheduler, pipe.scheduler.config)

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
        #pipe.enable_sequential_cpu_offload()
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
    else:
        pipe = pipe.to(f"cuda:{local_rank}")
    if args.xformers_efficient:
        pipe.enable_xformers_memory_efficient_attention()

    # warmup run
    start_time = time.time()
    output = pipe(
        height=512,
        width=512,
        prompt="a dog",
        num_inference_steps=30,
        guidance_scale=7,
        generator=torch.Generator(device="cpu").manual_seed(1),
        max_sequence_length=256,
        output_type="pil",
        use_resolution_binning=input_config.use_resolution_binning,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    torch.cuda.empty_cache()
    logger.info("Model initialization completed")
    initialized = True

    return


def generate_image_parallel(
    positive_prompt,
    negative_prompt,
    width,
    height,
    num_inference_steps,
    cfg,
    seed,
    clip_skip,
    output_path
):
    global pipe, local_rank, input_config
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
        f"output_path={output_path}\n"
    )
    logger.info(f"Starting image generation with prompt: {positive_prompt}")
    logger.info(f"Negative: {negative_prompt}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    args = get_args()

    match args.pipeline_type:
        case _:
            is_image = True
            output = pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                clip_skip=clip_skip,
                max_sequence_length=256,
                output_type="pil",
                use_resolution_binning=input_config.use_resolution_binning,
            )
    end_time = time.time()
    elapsed_time = end_time - start_time

    torch.cuda.empty_cache()

    # only last rank will output an image
    # this can be a bit race-y
    if pipe.is_dp_last_group():
        logger.info(f"Output generated on rank {local_rank}")
        if is_image:
            output.images[0].save(output_path)
        else:
            logging.info("Frames still need to be implemented!")
    else:
        logger.info(f"Rank {local_rank} task completed")
        # adjust delay as needed
        time.sleep(5)

    if is_image:
        image = Image.open(output_path)
        pickled = pickle.dumps(image)
        output_base64 = base64.b64encode(pickled).decode("utf-8")
        return output_base64, elapsed_time, is_image
    else:
        logging.info("Frames still need to be implemented!")
        return None, elapsed_time, is_image


@app.route("/generate", methods=["POST"])
def generate_image():
    logger.info("Received POST request for image generation")
    data = request.json
    positive_prompt     = data.get("positive_prompt", None)
    negative_prompt     = data.get("negative_prompt", None)
    width               = data.get("width")
    height              = data.get("height")
    num_inference_steps = data.get("num_inference_steps")
    cfg                 = data.get("cfg",)
    seed                = data.get("seed")
    clip_skip           = data.get("clip_skip")
    output_path         = data.get("output_path", None)

    assert output_path is not None, "No output path provided"

    logger.info(
        "Request parameters:\n"
        f"positive_prompt='{positive_prompt}'\n"
        f"negative_prompt='{negative_prompt}'\n"
        f"width={width}\n"
        f"height={height}\n"
        f"steps={num_inference_steps}\n"
        f"cfg={cfg}\n"
        f"seed={seed}\n"
        f"clip_skip={clip_skip}\n"
        f"output_path={output_path}\n"
    )

    # Broadcast request parameters to all processes
    params = [positive_prompt, negative_prompt, width, height, num_inference_steps, cfg, seed, clip_skip, output_path]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    output_base64, elapsed_time, is_image = generate_image_parallel(*params)

    response = {
        "elapsed_time": f"{elapsed_time:.2f} sec",
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
            params = [None] * 9 # len(params) of generate_image_parallel()
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
