#!/usr/bin/env python3
import argparse
import backend_pb2
import backend_pb2_grpc
import base64
import grpc
import json
import logging
import os
import pickle
import requests
import signal
import subprocess
import sys
import time
import torch
from concurrent import futures
from diffusers.utils import export_to_video, export_to_gif


#os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
#os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
#os.environ["NCCL_SHM_DISABLE"] = "1"
#os.environ["NCCL_DEBUG"] = "INFO"
#os.environ["NCCL_P2P_DISABLE"] = "0"


config              =   json.load(open("multigpu_diffusion/config.json"))
config_options      =   {}
"""
    GENERIC OPTIONS
        variant
        compel
        compile_unet
        compile_vae
        compile_text_encoder
        quantize_encoder
        quantize_encoder_type
        enable_vae_tiling
        enable_vae_slicing
        xformers_efficient
        warm_up_steps

    ASYNCDIFF OPTIONS
        time_shift
        model_n
        stride

    DISTRIFUSER OPTIONS
        sync_mode
        parallelism
        split_scheme
        no_split_batch
        no_cuda_graph

    XDIT OPTIONS
        pipefusion_parallel_degree
        tensor_parallel_degree
        data_parallel_degree
        ulysses_degree
        ring_degree
        use_cfg_parallel
"""
NON_ARG_OPTIONS = [
    "port",
    "grpc_port",
    "master_port",
    "cuda_devices",
    "nproc_per_node",
    "backend",
    "frames",
    "video_output_type",
    "chunk_size",
    "motion_bucket_id",
    "noise_aug_strength",
    "scheduler_args",
]


port = 6000
master_port = 29400
grpc_port = 50050
host_init_timeout = 1800


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))


process = None
def kill_process():
    global process
    if process is not None:
        process.kill()
        time.sleep(3)
        process = None


def setup_logger():
    global logger
    logging.basicConfig(
        level=logging.INFO,
        format=f"[LocalAI Backend] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)


# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    last = {
        "height": 512,
        "width": 512,
        "model": None,
        "type": None,
        "cfg_scale": 7,
        "variant": "fp16",
        "scheduler": None,
        "loras": None,
        "controlnet": None,
        "clip_skip": 0,
        "low_vram": False,
    }

    # class vars
    loaded = False
    needs_reload = False
    process_type = ""
    nproc_per_node = torch.cuda.device_count()


    def log_reload_reason(self, reason):
        self.needs_reload = True
        logger.info("Pipeline needs reload: " + reason)


    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))


    def LoadModel(self, request, context):
        options = request.Options
        for opt in options:
            if ":" in opt:
                o = opt.split(":")
                config_options[o[0]] = opt.replace(f"{o[0]}:", "", 1)
            else:
                config_options[opt] = ""

        if config_options.get("port") is not None:
            global port
            port = int(config_options["port"])
        if config_options.get("master_port") is not None:
            global master_port
            master_port = int(config_options["master_port"])
        if config_options.get("cuda_devices") is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = config_options["cuda_devices"]
        if config_options.get("nproc_per_node") is not None:
            self.nproc_per_node = int(config_options["nproc_per_node"])

        if config_options.get("backend"):
            match config_options["backend"]:
                case "asyncdiff":
                    if self.process_type != "asyncdiff":
                        self.log_reload_reason(f"Host type changed to AsyncDiff with {self.nproc_per_node} GPUs")
                        self.process_type = "asyncdiff"
                case "distrifuser":
                    if self.process_type != "distrifuser":
                        self.log_reload_reason(f"Host type changed to DistriFuser with {self.nproc_per_node} GPUs")
                        self.process_type = "distrifuser"
                case "xdit":
                    if self.process_type != "xdit":
                        self.log_reload_reason(f"Host type changed to xDiT with {self.nproc_per_node} GPUs")
                        self.process_type = "xdit"
                case _:
                    assert False, f"Unknown backend: {config_options['backend']}"
        else:
            # detect host to use
            if self.last["type"] in ["sd1", "sd2", "sdxl"] and not (self.nproc_per_node > 2 and request.LoraAdapters):
                # distrifuser
                if self.process_type != "distrifuser":
                    self.log_reload_reason(f"Host type changed to DistriFuser with {self.nproc_per_node} GPUs")
                    self.process_type = "distrifuser"
            elif self.last["type"] in ["flux", "sd3"]:
                # xdit
                if self.process_type != "xdit":
                    self.log_reload_reason(f"Host type changed to xDiT with {self.nproc_per_node} GPUs")
                    self.process_type = "xdit"
            elif self.last["type"] in ["ad", "sd1", "sd2", "sd3", "sdup", "sdxl", "svd"]:
                # asyncdiff
                if self.process_type != "asyncdiff":
                    self.log_reload_reason(f"Host type changed to AsyncDiff with {self.nproc_per_node} GPUs")
                    self.process_type = "asyncdiff"
            else:
                assert False, f"Unsupported pipeline type: {request.PipelineType}"

        if request.PipelineType != self.last["type"]:
            self.log_reload_reason("Pipeline type changed")
        self.last["type"] = request.PipelineType

        if request.CFGScale > 0 and self.last["cfg_scale"] != request.CFGScale:
            self.last["cfg_scale"] = request.CFGScale

        if request.Model != self.last["model"] or (len(request.ModelFile) > 0 and os.path.exists(request.ModelFile) and request.ModelFile != self.last["model"]):
            self.log_reload_reason("Model changed")

        self.last["model"] = request.Model
        if request.ModelFile != "" and os.path.exists(request.ModelFile):
            self.last["model"] = request.ModelFile
            self.log_reload_reason("Model changed")

        if config_options.get("variant") is not None:
            if config_options["variant"] != self.last["variant"]:
                if config_options["variant"] == "fp16":    self.last["variant"] = "fp16"
                elif config_options["variant"] == "bf16":  self.last["variant"] = "bf16"
                elif config_options["variant"] == "fp32":  self.last["variant"] = "fp32"
                else: assert False, f"Unsupported torch memory dtype: {config_options['variant']}"
                self.log_reload_reason("Pipeline memory type changed")
        else:
            if self.last["variant"] != "fp16":
                self.last["variant"] = "fp16"
                self.log_reload_reason("Pipeline memory type changed")

        if request.SchedulerType != self.last["scheduler"]:
            self.last["scheduler"] = { "scheduler": request.SchedulerType }
            self.log_reload_reason("Scheduler changed")

        if config_options.get("scheduler_args") is not None:
            if self.last.get("scheduler") is None:
                self.last["scheduler"] = {}
            scheduler_args = config_options.get("scheduler_args")
            for k,v in json.loads(scheduler_args).items():
                self.last["scheduler"][k] = v

        if request.LoraAdapters:
            if (self.last["loras"] is None or len(self.last["loras"].keys()) == 0) and len(request.LoraAdapters) == 0:
                pass
            elif self.last["loras"] is None and len(request.LoraAdapters) > 0 or len(self.last["loras"].keys()) != len(request.LoraAdapters):
                self.log_reload_reason("LoRAs changed")
            else:
                for adapter in self.last["loras"].keys():
                    if adapter not in request.LoraAdapters:
                        self.log_reload_reason("LoRAs changed")
                        break
            if self.needs_reload:
                self.last["loras"] = {}
                if len(request.LoraAdapters) > 0:
                    i = 0
                    for adapter in request.LoraAdapters:
                        self.last["loras"][adapter] = request.LoraScales[i]
                        i += 1

        if len(request.ControlNet) > 0 and request.ControlNet != self.last["controlnet"]:
            self.last["controlnet"] = request.ControlNet
            self.log_reload_reason("ControlNet changed")

        if self.last["clip_skip"] != request.CLIPSkip:
            self.last["clip_skip"] = request.CLIPSkip

        if self.last["low_vram"] != request.LowVRAM:
            self.last["low_vram"] = request.LowVRAM
            self.log_reload_reason("Low VRAM changed")

        return backend_pb2.Result(message="", success=True)


    def GenerateImage(self, request, context):
        if request.height != self.last["height"] or request.width != self.last["width"]:
            self.log_reload_reason("Resolution changed")
            self.last["height"] = request.height
            self.last["width"] = request.width

        if not self.loaded or self.needs_reload:
            if self.process_type == "asyncdiff":
                logging.info("Using AsyncDiff host for pipeline type: " + self.last["type"])
                assert self.nproc_per_node > 1, "AsyncDiff requires at least 2 GPUs."
                assert self.nproc_per_node < 6, "AsyncDiff does not support more than 5 GPUs. You can set a limit using CUDA_VISIBLE_DEVICES."
                self.launch_host("asyncdiff")

            elif self.process_type == "distrifuser":
                logging.info("Using DistriFuser host for pipeline type: " + self.last["type"])
                self.launch_host("distrifuser")

            elif self.process_type == "xdit":
                logging.info("Using xDiT host for pipeline type: " + self.last["type"])
                self.launch_host("xdit")

        if self.loaded:
            url = f"http://localhost:{port}/generate"

            if self.last["type"] in ["svd"]:
                data = {
                    "image":                request.src,
                    "decode_chunk_size":    int(config_options["chunk_size"]),
                    "frames":               int(config_options["frames"]),
                    "motion_bucket_id":     int(config_options["motion_bucket_id"]),
                    "noise_aug_strength":   float(config_options["noise_aug_strength"]),
                }

            elif self.last["type"] in ["sdup"]:
                data = {
                    "image":                request.src,
                }

            else:
                data = {
                    "clip_skip":            int(self.last["clip_skip"]),
                }

            data["cfg"]     = float(self.last["cfg_scale"])
            data["seed"]    = int(request.seed)
            data["steps"]   = int(request.step)
            if request.positive_prompt is not None: data["positive"] = request.positive_prompt
            if request.negative_prompt is not None: data["negative"] = request.negative_prompt

            logger.info("Sending request to {url} with params:\n" + str(data))
            try:
                response = requests.post(url, json=data)
                response_data = response.json()
            except:
                kill_process()
                self.log_reload_reason("Invalid or no response from host (Failed to send request/parse response)")
                return backend_pb2.Result(message="No image generated", success=False)

            output_base64 = response_data.get("output")
            if output_base64 is not None:
                output_bytes = base64.b64decode(output_base64)
                output = pickle.loads(output_bytes)
            else:
                kill_process()
                self.log_reload_reason("Invalid or no response from host (No image received)")
                return backend_pb2.Result(message="No image generated", success=False)

            if output is not None:
                if response_data.get("is_image"):
                    output.save(request.dst)
                else:
                    if config_options["video_output_type"].lower() == "mp4":
                        file_name = request.dst.replace(".png", ".mp4")
                        logger.info("Exporting frames to MP4")
                        export_to_video(output, file_name, fps=7)
                    elif config_options["video_output_type"].lower() == "gif":
                        file_name = request.dst.replace(".png", ".gif")
                        logger.info("Exporting frames to GIF")
                        export_to_gif(output, file_name)
                    else:
                        logger.info("Unsupported or no media type given")
                        return backend_pb2.Result(message="Media created but not saved", success=True)
                return backend_pb2.Result(message="Media generated", success=True)
            else:
                return backend_pb2.Result(message="No image generated", success=False)
        else:
            return backend_pb2.Result(message="Host is not loaded", success=False)


    def launch_host(self, name):
        global process, config_options, NON_ARG_OPTIONS
        cmd = ['torchrun', f'--nproc_per_node={self.nproc_per_node}', f'--master-port={master_port}', f'multigpu_diffusion/host_{name}.py']

        # generic
        cmd.append(f'--port={port}')
        cmd.append(f'--checkpoint={self.last["model"]}')
        cmd.append(f'--type={self.last["type"]}')
        cmd.append(f'--height={self.last["height"]}')
        cmd.append(f'--width={self.last["width"]}')
        for k, v in config_options.items():
            if k not in NON_ARG_OPTIONS:
                if len(v) > 0:  cmd.append(f'--{k}={v}')
                else:           cmd.append(f'--{k}')

        if self.last["controlnet"] is not None:                                                                 cmd.append(f'--control_net={self.last["controlnet"]}')
        if self.last["loras"] is not None and len(self.last["loras"]) > 0:                                      cmd.append(f'--lora={json.dumps(self.last["loras"])}')
        if self.last["scheduler"] is not None and self.last["type"] in ['sd1', 'sd2', 'sd3', 'sdup', 'sdxl']:   cmd.append(f'--scheduler={json.dumps(self.last["scheduler"])}')

        kill_process()

        global process
        logger.info("Starting torch instance:\n" + str(cmd))
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
        initialize_url = f"http://localhost:{port}/initialize"
        time_elapsed = 0
        while True:
            time.sleep(5)
            time_elapsed += 5
            try:
                response = requests.get(initialize_url)
                if response.status_code == 200:
                    self.loaded = True
                    self.needs_reload = False
                    logger.info("Torch instance started successfully")
                    return
            except requests.exceptions.RequestException:
                if time_elapsed > host_init_timeout:
                    kill_process()
                    self.log_reload_reason(f'Torch instance start has timed out ({host_init_timeout}s)')
                    return backend_pb2.Result(message=f'Failed to launch host within {host_init_timeout} seconds', success=False)
                else:
                    logger.info(f'Waiting on torch instance init ({time_elapsed}s/{host_init_timeout}s)')


def serve(address):
    setup_logger()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    server.add_insecure_port(address)
    server.start()
    logger.info("Server started. Listening on: " + address)

    # Define the signal handler function
    def signal_handler(sig, frame):
        logger.info("Received termination signal. Shutting down...")
        kill_process()
        server.stop(0)
        sys.exit(0)

    # Set the signal handlers for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except:
        kill_process()
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument("--addr", default=f'localhost:{grpc_port}', help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)
