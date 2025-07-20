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
from pathlib import Path
from PIL import Image


os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
#os.environ["NCCL_SHM_DISABLE"] = "1"
#os.environ["NCCL_DEBUG"] = "INFO"
#os.environ["NCCL_P2P_DISABLE"] = "0"


config              = json.load(open("config.json"))
config_global       = config["global"]
config_asyncdiff    = config["asyncdiff"]
config_distrifuser  = config["distrifuser"]
config_xdit         = config["xdit"]
config_extra        = {}
if len(config_global["cuda_devices"]) > 0:  os.environ["CUDA_VISIBLE_DEVICES"] = config_global["cuda_devices"]


URL = f'http://localhost:{config_global["port"]}'


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))


process = None
def kill_process():
    global process
    if process is not None:
        process.terminate()
        time.sleep(15)
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
    # pipeline vars
    height = None
    width = None
    model = None
    cfg_scale = 7
    type = None
    scheduler = None
    variant = "fp16"
    loras = {}
    controlnet = None
    clip_skip = 0
    low_vram = False

    # class vars
    loaded = False
    needs_reload = False
    process_type = -1 # 0 = asyncdiff, 1 = distrifuser, 2 = xDiT
    nproc_per_node = torch.cuda.device_count()


    def log_reload_reason(self, reason):
        self.needs_reload = True
        logger.info("Pipeline needs reload: " + reason)


    def Health(self, request, context):
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))


    def LoadModel(self, request, context):
        # TODO: use options
        options = request.Options
        for opt in options:
            o = opt.split(":")
            assert len(o) == 2, "Malformed option: " + opt
            k = o[0]
            v = o[1]
            config_extra[k] = v

        # detect host to use
        self.type = request.PipelineType
        if self.type in ["sd1", "sd2", "sdxl"] and self.nproc_per_node <= 2:
            # distrifuser
            if self.process_type != 1:
                self.log_reload_reason(f"Host type changed to DistriFuser with {self.nproc_per_node} GPUs")
                self.process_type = 1
        elif self.type in ["flux", "sd3"]:
            # xdit
            if self.process_type != 2:
                self.log_reload_reason(f"Host type changed to xDiT with {self.nproc_per_node} GPUs")
                self.process_type = 2
        elif self.type in ["ad", "sd1", "sd2", "sd3", "sdup", "sdxl", "svd"]:
            # asyncdiff
            if self.process_type != 0:
                self.log_reload_reason(f"Host type changed to AsyncDiff with {self.nproc_per_node} GPUs")
                self.process_type = 0
        else:
            assert False, "Unsupported pipeline type"

        if request.PipelineType != self.type: 
            self.type = request.PipelineType
            self.log_reload_reason("Pipeline type changed")

        if request.CFGScale > 0 and self.cfg_scale != request.CFGScale:
            self.cfg_scale = request.CFGScale

        if request.Model != self.model or (len(request.ModelFile) > 0 and os.path.exists(request.ModelFile) and request.ModelFile != self.model):
            self.log_reload_reason("Model changed")

        self.model = request.Model
        if request.ModelFile != "" and os.path.exists(request.ModelFile):
            self.model = request.ModelFile
            self.log_reload_reason("Model changed")

        if request.F16Memory and self.variant != "fp16":
            self.variant = "fp16"
            self.log_reload_reason("Pipeline memory type changed")
        elif not request.F16Memory and self.variant != "fp32":
            self.variant = "fp32"
            self.log_reload_reason("Pipeline memory type changed")

        if request.SchedulerType != self.scheduler:
            self.scheduler = request.SchedulerType
            self.log_reload_reason("Scheduler changed")

        if request.LoraAdapters:
            if len(self.loras.keys()) == 0 and len(request.LoraAdapters) == 0:
                pass
            elif len(self.loras.keys()) != len(request.LoraAdapters):
                self.log_reload_reason("LoRAs changed")
            else:
                for adapter in self.loras.keys():
                    if adapter not in request.LoraAdapters:
                        self.log_reload_reason("LoRAs changed")
                        break
            if self.needs_reload:
                self.loras = {}
                if len(request.LoraAdapters) > 0:
                    i = 0
                    for adapter in request.LoraAdapters:
                        self.loras[adapter] = request.LoraScales[i]
                        i += 1

        if len(request.ControlNet) > 0 and request.ControlNet != self.controlnet:
            self.controlnet = request.ControlNet
            self.log_reload_reason("ControlNet changed")

        if self.clip_skip != request.CLIPSkip:
            self.clip_skip = request.CLIPSkip

        if self.low_vram != request.LowVRAM:
            self.low_vram = request.LowVRAM
            self.log_reload_reason("Low VRAM changed")

        return backend_pb2.Result(message="", success=True)


    def GenerateImage(self, request, context):
        if request.height != self.height or request.width != self.width:
            self.log_reload_reason("Resolution changed")
            self.height = request.height
            self.width = request.width

        if not self.loaded or self.needs_reload:
            if self.process_type == 0:
                logging.info("Using AsyncDiff host for pipeline type: " + self.type)

                assert self.nproc_per_node > 1, "AsyncDiff requires at least 2 GPUs."
                assert self.nproc_per_node < 6, "AsyncDiff does not support more than 5 GPUs. You can set a limit using CUDA_VISIBLE_DEVICES."
                match self.nproc_per_node:
                    case 2:
                        model_n = 2
                        stride = 1
                    case 3:
                        model_n = 2
                        stride = 2
                    case 4:
                        model_n = 3
                        stride = 2
                    case 5:
                        # untested
                        model_n = 4
                        stride = 2

                config = {}
                config["model_n"] = model_n
                config["stride"] = stride
                self.launch_host(config, "asyncdiff")

            elif self.process_type == 1:
                logging.info("Using DistriFuser host for pipeline type: " + self.type)

                config = {}
                self.launch_host(config, "distrifuser")

            elif self.process_type == 2:
                logging.info("Using xDiT host for pipeline type: " + self.type)
                host = "host_xdit.py"

                assert self.nproc_per_node > 1, "xDiT requires at least 2 GPUs."
                """
                    Support for parallelism techniques (as per xDiT README, at the time of writing):
                        - Tensor Parallel:
                            - StepVideo
                        - PipeFusion:
                            - Hunyuan
                            - Flux
                            - PixArt
                            - SD3
                        - CFG Parallel:
                            - ConsisID
                            - CogVideoX
                            - Mochi
                            - Hunyuan
                            - PixArt
                            - SD3
                            - SDXL
                """

                pf_deg = config_xdit["pipefusion_parallel_degree"]
                tp_deg = config_xdit["tensor_parallel_degree"]
                dp_deg = config_xdit["data_parallel_degree"]
                u_deg = config_xdit["ulysses_degree"]
                r_deg = config_xdit["ring_degree"]
                cfg_parallel = 2 if config_xdit["use_cfg_parallel"] else 1
                test_product = pf_deg * tp_deg * dp_deg * u_deg * r_deg * cfg_parallel
                assert test_product == self.nproc_per_node, "pipefusion * tensor * data * ulysses * ring * (cfg ? 2 : 1) must equal nproc_per_node."

                config = {}
                self.launch_host(config, "xdit")

        if self.loaded:
            url = f"{URL}/generate"

            if self.type in ["sdup", "svd"]:
                if self.low_vram and config_asyncdiff["chunk_size"] >= 2:
                    decode_chunk_size = 2
                else:
                    decode_chunk_size = config_asyncdiff["chunk_size"]

                data = {
                    "image": request.src,
                    "steps": request.step,
                    "seed": request.seed,
                    "cfg": self.cfg_scale,
                    "decode_chunk_size": decode_chunk_size,
                    "frames": config_asyncdiff["frames"],
                    "motion_bucket_id": config_asyncdiff["motion_bucket_id"],
                    "noise_aug_strength": config_asyncdiff["noise_aug_strength"],
                    "output_path": request.dst,
                }

                if request.positive_prompt and len(request.positive_prompt) > 0:
                    data["positive"] = request.positive_prompt

            else:
                data = {
                    "positive": request.positive_prompt,
                    "steps": request.step,
                    "seed": request.seed,
                    "cfg": self.cfg_scale,
                    "clip_skip": self.clip_skip,
                    "output_path": request.dst,
                }

            if request.negative_prompt and len(request.negative_prompt) > 0:
                data["negative"] = request.negative_prompt

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
                    if config_asyncdiff["video_output_type"].lower() == "mp4":
                        file_name = request.dst.replace(".png", ".mp4")
                        logger.info("Exporting frames to MP4")
                        export_to_video(output, file_name, fps=7)
                    elif config_asyncdiff["video_output_type"].lower() == "gif":
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


    def launch_host(self, config, name):
        global process, config_asyncdiff, config_distrifuser, config_xdit, config_global
        cmd = ['torchrun', f'--nproc_per_node={self.nproc_per_node}', f'--master-port={config_global["master_port"]}', f'host_{name}.py']

        for k,v in config.items(): cmd.append(f'--{k}={v}')

        # asyncdiff
        if name == "asyncdiff":
            cmd.append(f'--time_shift={config_asyncdiff["time_shift"]}')

        # distrifuser
        elif name == "distrifuser":
            cmd.append(f'--sync_mode={config_distrifuser["sync_mode"]}')
            cmd.append(f'--parallelism={config_distrifuser["parallelism"]}')
            cmd.append(f'--split_scheme={config_distrifuser["split_scheme"]}')
            if config_distrifuser["no_split_batch"]:    cmd.append('--no_split_batch')
            if config_distrifuser["no_cuda_graph"]:     cmd.append('--no_cuda_graph')

        # xdit
        elif name == "xdit":
            cmd.append(f'--pipefusion_parallel_degree={config_xdit["pipefusion_parallel_degree"]}')
            cmd.append(f'--tensor_parallel_degree={config_xdit["tensor_parallel_degree"]}')
            cmd.append(f'--data_parallel_degree={config_xdit["data_parallel_degree"]}')
            cmd.append(f'--ulysses_degree={config_xdit["ulysses_degree"]}')
            cmd.append(f'--ring_degree={config_xdit["ring_degree"]}')
            cmd.append('--no_use_resolution_binning')
            if config_xdit["use_cfg_parallel"]:         cmd.append("--use_cfg_parallel")

        # generic
        cmd.append(f'--port={config_global["port"]}')
        cmd.append(f'--model={self.model}')
        cmd.append(f'--type={self.type}')
        cmd.append(f'--variant={self.variant}')
        cmd.append(f'--height={self.height}')
        cmd.append(f'--width={self.width}')
        for k,v in config_extra.items():                cmd.append(f'--{k}={v}')

        if self.controlnet is not None:                 cmd.append(f'--controlnet={self.controlnet}')

        if self.low_vram:
            cmd.append('--enable_vae_tiling')
            cmd.append('--enable_vae_slicing')
            cmd.append('--xformers_efficient')

        if config_global["compile_unet"]:               cmd.append('--compile_unet')
        if config_global["compile_vae"]:                cmd.append('--compile_vae')
        if config_global["compile_text_encoder"]:       cmd.append('--compile_text_encoder')

        if config_global["quantize_encoder"]:
            cmd.append("--quantize_encoder")
            cmd.append(f'--quantize_encoder_type={config_global["quantize_encoder_type"]}')

        cmd.append(f'--warm_up_steps={config_global["warm_up_steps"]}')

        if config_global["use_compel"] and self.process_type != 2:
            cmd.append('--compel')

        if len(self.loras) > 0:
            cmd.append(f'--lora={json.dumps(self.loras)}')

        if self.scheduler is not None and len(self.scheduler) > 0 and self.type in ['sd1', 'sd2', 'sd3', 'sdup', 'sdxl']:
            cmd.append(f'--scheduler={self.scheduler}')

        kill_process()

        global process
        logger.info("Starting torch instance:\n" + str(cmd))
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=subprocess.STDOUT)
        initialize_url = f"{URL}/initialize"
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
                if time_elapsed > config_global["host_init_timeout"]:
                    kill_process()
                    self.log_reload_reason('Torch instance start has timed out (config_global["host_init_timeout"]s)')
                    return backend_pb2.Result(message=f'Failed to launch host within {config_global["host_init_timeout"]} seconds', success=False)
                else:
                    logger.info(f'Waiting on torch instance init ({time_elapsed}s/{config_global["host_init_timeout"]}s)')


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
    parser.add_argument("--addr", default=f'localhost:{config_global["grpc_port"]}', help="The address to bind the server to.")
    args = parser.parse_args()
    serve(args.addr)
