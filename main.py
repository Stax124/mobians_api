import asyncio
import base64
import contextlib
import gc
import hashlib
import io
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Optional
from uuid import uuid4

import numpy as np
import PIL.ImageOps
import redis
import tomesd
import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionGLIGENPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hyper_tile import split_attention
from PIL import Image
from pydantic import BaseModel
from redis.backoff import ExponentialBackoff
from redis.exceptions import BusyLoadingError, ConnectionError, TimeoutError
from redis.retry import Retry

# from sfast.compilers.stable_diffusion_pipeline_compiler import (
#     CompilationConfig,
#     compile,
# )
from lpw_pipeline import StableDiffusionLongPromptWeightingPipeline
from submodules.DeepCache.DeepCache import DeepCacheSDHelper

torch.backends.cuda.matmul.allow_tf32 = True

# Run 3 retries with exponential backoff strategy
retry = Retry(ExponentialBackoff(), 3)

r = redis.Redis(
    host="76.157.184.213",
    port=6379,
    db=0,
    retry=retry,
    retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError],
)


class ImageRequestModel(BaseModel):
    prompt: str

    image: Optional[str] = None
    mask_image: Optional[str] = None
    control_image: Optional[str] = None
    scheduler: int
    steps: int
    negative_prompt: str
    width: int
    height: int
    guidance_scale: int
    seed: int
    batch_size: int
    strength: Optional[float] = None
    job_type: str
    model: Optional[str] = None
    fast_pass_enabled: Optional[bool] = False
    deepcache: bool = False
    hypertile: bool = False


executor = ThreadPoolExecutor(max_workers=5)
app = FastAPI()
jobs = {}


# Set up the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


def free_memory():
    # Free up memory
    torch.cuda.empty_cache()


def load_model():
    global helper

    # NOTE: You could change to StableDiffusionXLPipeline to load SDXL model
    # model = StableDiffusionXLPipeline.from_pretrained(
    #     'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16)
    model: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(  # type: ignore
        r"./sonicdiffusion_v4.safetensors",
        torch_dtype=torch.float16,
        # custom_pipeline="lpw_stable_diffusion",
        use_safetensors=True,
    )

    model.scheduler = EulerAncestralDiscreteScheduler.from_config(
        model.scheduler.config
    )
    model.safety_checker = None
    model.enable_vae_slicing()
    model.enable_vae_tiling()

    # model.load_lora_weights('add_detail.safetensors')
    # model.fuse_lora()
    model.to(torch.device("cuda"))
    # model.to(torch.device('cuda:1'))

    helper = DeepCacheSDHelper(pipe=model)
    helper.set_params(
        cache_interval=2,
        cache_branch_id=0,
    )

    return model, helper


stable_diffusion_txt2img, helper = load_model()

stable_diffusion_txt2img.unet.set_attn_processor(AttnProcessor2_0())
# stable_diffusion_txt2img.load_textual_inversion("EasyNegativeV2.safetensors")
# stable_diffusion_txt2img.load_textual_inversion(
#     "./embeddings/bad_prompt_version2-neg.pt"
# )
# stable_diffusion_txt2img.load_textual_inversion("./embeddings/BadDream.pt")
# stable_diffusion_txt2img.load_textual_inversion("./embeddings/boring_e621_v4.pt")
# stable_diffusion_txt2img.load_textual_inversion(
#     "./embeddings/By bad artist -neg-anime.pt"
# )
# stable_diffusion_txt2img.load_textual_inversion("./embeddings/By bad artist -neg.pt")
# stable_diffusion_txt2img.load_textual_inversion("./embeddings/deformityv6.pt")
# stable_diffusion_txt2img.load_textual_inversion("./embeddings/ERA09NEGV2.pt")
# stable_diffusion_txt2img.load_textual_inversion("./embeddings/fcDetailPortrait.pt")
# stable_diffusion_txt2img.load_textual_inversion("./embeddings/fcNeg-neg.pt")
# stable_diffusion_txt2img.load_textual_inversion("./embeddings/fluffynegative.pt")
# stable_diffusion_txt2img.load_textual_inversion(
#     "./embeddings/Unspeakable-Horrors-Composition-4v.pt"
# )
# stable_diffusion_txt2img.load_textual_inversion(
#     "./embeddings/verybadimagenegative_v1.3.pt"
# )

# tomesd.apply_patch(stable_diffusion_txt2img, ratio=0.3)

print("\nLoading Main Diffusion model")
# stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained("SonicDiffusionV4",
#                                         custom_pipeline="lpw_stable_diffusion",
#                                         torch_dtype=torch.float16,
#                                         revision="fp16",
#                                         safety_checker=None,
#                                         load_safety_checker=False,
#                                         feature_extractor=None,
#                                         requires_safety_checker=False,
#                                         use_safetensors=True,
#                                         ).to("cuda")

# stable_diffusion_txt2img.unet = torch.compile(stable_diffusion_txt2img.unet, mode="reduce-overhead")

# stable_diffusion_txt2img.enable_vae_slicing()
# stable_diffusion_txt2img.unet.set_attn_processor(AttnProcessor2_0())
# stable_diffusion_txt2img.load_textual_inversion("EasyNegativeV2.safetensors")
# stable_diffusion_txt2img.load_textual_inversion("OverallDetail.pt")
# tomesd.apply_patch(stable_diffusion_txt2img, ratio=0.3)

# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# stable_diffusion_txt2img.scheduler = EulerAncestralDiscreteScheduler.from_config(stable_diffusion_txt2img.scheduler.config, torch_dtype=torch.float16)

# # pipe.load_lora_weights('more_details.safetensors')
# print("Done loading Main Diffusion model")

# config = CompilationConfig.Default()
# xformers and Triton are suggested for achieving best performance.
# It might be slow for Triton to generate, compile and fine-tune kernels.
# try:
#     import xformers

#     config.enable_xformers = True
# except ImportError:
#     print("xformers not installed, skip")
# # NOTE: On some recent GPUs (for example, RTX 4080), Triton might generate slow kernels.
# # Disable Triton if you encounter this problem.
# try:
#     import triton

#     config.enable_triton = True
# except ImportError:
#     print("Triton not installed, skip")
# NOTE:
# CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
# My implementation can handle dynamic shape with increased need for GPU memory.
# But when your GPU VRAM is insufficient or the image resolution is high,
# CUDA Graph could cause less efficient VRAM utilization and slow down the inference.
# If you meet problems related to it, you should disable it.
# config.enable_cuda_graph = False
# config.preserve_parameters = False

# compiled_model = compile(stable_diffusion_txt2img, config)

kwarg_inputs = dict(
    prompt="(masterpiece:1,2), best quality, masterpiece, best detail face, lineart, monochrome, a beautiful girl",
    # NOTE: If you use SDXL, you should use a higher resolution to improve the generation quality.
    height=512,
    width=512,
    num_inference_steps=30,
    num_images_per_prompt=1,
)

# NOTE: Warm it up.
# The first call will trigger compilation and might be very slow.
# After the first call, it should be very fast.
# output_image = compiled_model(**kwarg_inputs).images[0]

# # Let's see the second call!
# output_image = compiled_model(**kwarg_inputs).images[0]

# lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
# stable_diffusion_txt2img.load_lora_weights(lcm_lora_id)
# stable_diffusion_txt2img.scheduler = LCMScheduler.from_config(stable_diffusion_txt2img.scheduler.config)
# stable_diffusion_txt2img.to(device="cuda", dtype=torch.float16)

# img2img
print("\n Loading Img2Img model")
components = stable_diffusion_txt2img.components
components["safety_checker"] = None
stable_diffusion_txt2img = StableDiffusionLongPromptWeightingPipeline(
    **components, requires_safety_checker=False
)
# stable_diffusion_txt2img.to(torch.device('cuda'))
# stable_diffusion_txt2img.to(torch.device('cuda:1'))
print("Done loading Img2Img model")


# # inpaint
# print("\nLoading Inpainting model")
# inpainting = StableDiffusionInpaintPipeline.from_single_file(
#     pretrained_model_link_or_path=r"./SonicDiffusionV4-inpainting.inpainting.safetensors",
#     torch_dtype=torch.float16,
#     revision="fp16",
#     safety_checker=None,
#     feature_extractor=None,
#     requires_safety_checker=False,
#     # use_safetensors=True,
#     cache_dir="",
#     load_safety_checker=False,
# ).to("cuda")
# inpainting.scheduler = EulerAncestralDiscreteScheduler.from_config(
#     inpainting.scheduler.config
# )
# inpainting.to(torch.device("cuda"))
# # inpainting.to(torch.device('cuda:1'))
# # inpainting.enable_vae_slicing()
# # inpainting.enable_model_cpu_offload()
# tomesd.apply_patch(inpainting, ratio=0.3)

# inpainting.unet = torch.compile(stable_diffusion_txt2img.unet, mode="reduce-overhead", fullgraph=True, dynamic=True)

print("Done loading Inpainting model")

# Controlnet
# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/control_v11p_sd15_inpaint",
#     torch_dtype=torch.float16
# ).to("cuda")
# inpainting = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#     "sonicFluffy_trainDiff_75_25", controlnet=controlnet,
#     torch_dtype=torch.float16
# ).to("cuda")
# inpainting.scheduler = EulerAncestralDiscreteScheduler.from_config(inpainting.scheduler.config)
# #inpainting.scheduler = UniPCMultistepScheduler.from_config(inpainting.scheduler.config)
# #inpainting.scheduler = DDIMScheduler.from_config(inpainting.scheduler.config)
# inpainting.enable_vae_slicing()
# tomesd.apply_patch(inpainting, ratio=0.3)

# Controlnet tile
# controlnet = ControlNetModel.from_pretrained(
#     "lllyasviel/control_v11f1e_sd15_tile",
#     torch_dtype=torch.float16
# ).to("cuda")
# tile = DiffusionPipeline.from_pretrained(
#     "testSonicBeta4",
#     controlnet=controlnet,
#     custom_pipeline="stable_diffusion_controlnet_img2img",
#     torch_dtype=torch.float16
# ).to("cuda")
# tile.scheduler = EulerAncestralDiscreteScheduler.from_config(tile.scheduler.config)
# tile.enable_vae_slicing()
# tomesd.apply_patch(tile, ratio=0.3)

# gligen = StableDiffusionGLIGENPipeline.from_pretrained("testSonicBeta4",
#                                                         variant="fp16",
#                                                         torch_dtype=torch.float16
#                                                         ).to("cuda")


def conditional_context_manager(condition, context_manager):
    if condition:
        print("Context manager enabled")
        return context_manager
    else:
        print("Context manager disabled")
        return contextlib.nullcontext()


def process_image_task(request_data: ImageRequestModel, job_id, job_type):
    # Seed filtering is done on django so it doesnt crash the program (hopefully)
    seed = torch.Generator(device="cuda").manual_seed(request_data.seed)
    positive_prompt = clean_tags(request_data.prompt)
    negative_prompt = clean_tags(request_data.negative_prompt)

    if request_data.deepcache and not hasattr(helper, "function_dict"):
        helper.enable()
        print("Deepcache enabled")
    elif not request_data.deepcache and hasattr(helper, "function_dict"):
        helper.disable()
        print("Deepcache disabled")

    with torch.inference_mode():
        if job_type == "txt2img":
            with conditional_context_manager(
                request_data.hypertile,
                split_attention(
                    stable_diffusion_txt2img.unet,
                    aspect_ratio=request_data.width / request_data.height,
                    tile_size=256,
                ),
            ):
                try:
                    images = stable_diffusion_txt2img(
                        prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        num_images_per_prompt=4,
                        num_inference_steps=20,
                        width=request_data.width,
                        height=request_data.height,
                        guidance_scale=float(request_data.guidance_scale),
                        generator=seed,
                        # cross_attention_kwargs={"scale": 1}
                    ).images
                    for i, image in enumerate(images):
                        image.save(
                            f"{'hypertile-' if request_data.hypertile else ''}{'deepcache-' if request_data.deepcache else ''}{i}.png"
                        )
                except Exception as e:
                    raise e
                    print(e)
                    images = []
                    time_stamp = jobs[job_id]["timestamp"]
                    jobs[job_id] = {
                        "status": "failed",
                        "where": "txt2img",
                        "timestamp": time_stamp,
                    }

        elif job_type == "img2img":
            try:
                # Convert base64 string to PIL Image
                base64_image = request_data.image
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))

                images = stable_diffusion_txt2img.img2img(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    strength=float(request_data.strength),
                    num_images_per_prompt=4,
                    num_inference_steps=20,
                    guidance_scale=float(request_data.guidance_scale),
                    generator=seed,
                    # cross_attention_kwargs={"scale": 0.5}
                ).images

            except Exception as e:
                print(e)
                images = []
                time_stamp = jobs[job_id]["timestamp"]
                jobs[job_id] = {
                    "status": "failed",
                    "where": "img2img",
                    "timestamp": time_stamp,
                }

        elif job_type == "inpainting":
            try:
                # Convert base64 string to PIL Image
                base64_image = request_data.image
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))
                # image.show()

                # Same for inpaint_mask
                base64_mask_image = request_data.mask_image
                mask_image_data = base64.b64decode(base64_mask_image)
                mask_image = Image.open(io.BytesIO(mask_image_data))
                mask_image = PIL.ImageOps.invert(mask_image)
                # mask_image.show()

                # control_image = make_inpaint_condition(image, mask_image)
                # control_image.show()

                images = inpainting(
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    mask_image=mask_image,
                    strength=float(request_data.strength),
                    num_images_per_prompt=4,
                    num_inference_steps=20,
                    guidance_scale=float(request_data.guidance_scale),
                    generator=seed,
                    width=request_data.width,
                    height=request_data.height,
                    # control_image=control_image
                ).images

            except Exception as e:
                print(e)
                time_stamp = jobs[job_id]["timestamp"]
                jobs[job_id] = {
                    "status": "failed",
                    "where": "inpainting",
                    "timestamp": time_stamp,
                }

        elif job_type == "tile":
            try:
                # Convert base64 string to PIL Image
                base64_image = request_data.image
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))

                images = tile(
                    prompt=request_data.prompt,
                    negative_prompt=request_data.negative_prompt,
                    image=image,
                    controlnet_conditioning_image=image,
                    width=512,
                    height=512,
                    strength=1.0,
                    generator=torch.manual_seed(0),
                    num_inference_steps=32,
                ).images

            except Exception as e:
                print(e)
                images = []
                time_stamp = jobs[job_id]["timestamp"]
                jobs[job_id] = {
                    "status": "failed",
                    "where": "tile controlnet",
                    "timestamp": time_stamp,
                }

        elif job_type == "gligen":
            try:
                # Convert base64 string to PIL Image
                base64_image = request_data.image
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))

                images = gligen(
                    prompt=request_data.prompt,
                    negative_prompt=request_data.negative_prompt,
                    image=image,
                    width=512,
                    height=512,
                    strength=1.0,
                    generator=torch.manual_seed(0),
                    num_inference_steps=32,
                ).images

            except Exception as e:
                print(e)
                time_stamp = jobs[job_id]["timestamp"]
                jobs[job_id] = {
                    "status": "failed",
                    "where": "gligen",
                    "timestamp": time_stamp,
                }
        else:
            print("Invalid job type")
            return

        free_memory()

        # Convert PIL Image objects to base64 strings
        # images = [image_to_base64(img) for img in images]
        time_stamp = jobs[job_id]["timestamp"]
        jobs[job_id] = {
            "status": "completed",
            "result": images,
            "timestamp": time_stamp,
            "request_data": request_data,
        }


@app.websocket("/ws/queue_length")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Imagine get_queue_length() is a function that retrieves the current queue length
            queue_length = get_queue_length()
            await websocket.send_json({"queue_length": queue_length})
            await asyncio.sleep(5)  # Send update every second, for example
    except WebSocketDisconnect:
        # Handle disconnect, e.g., clean up resources, log, etc.
        print("Client disconnected from WebSocket.")


# @app.get("/get_queue_length/")
def get_queue_length():
    # Return queue length where jobs are either running or processing
    queue_length = 0
    for j_id, j in jobs.items():
        if j["status"] == "running" or j["status"] == "processing":
            queue_length += 1
    # return {"queue_length": queue_length}
    return queue_length


@app.post("/submit_job/")
async def submit_job(request: ImageRequestModel):
    job_id = str(uuid4())
    jobs[job_id] = {
        "status": "running",
        "request_data": request,
        "timestamp": time.time(),
        "fast_pass_enabled": request.fast_pass_enabled,
    }
    return JSONResponse({"job_id": job_id})


@app.get("/get_job/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "completed":
        finished_job = {"status": job["status"]}

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                pipe = r.pipeline()
                for i, image in enumerate(job["result"]):
                    byte_arr = io.BytesIO()
                    # image.save(byte_arr, format='PNG')
                    image.save(byte_arr, format="webp", quality=95)
                    image_data = byte_arr.getvalue()
                    pipe.set(f"job:{job_id}:image:{i}", image_data)

                    # Compute and store the checksum
                    checksum = hashlib.sha256(image_data).hexdigest()
                    pipe.set(f"job:{job_id}:image:{i}:checksum", checksum)

                    # Include metadata
                    metadata = job["request_data"].json()
                    pipe.set(f"job:{job_id}:metadata", metadata)
                pipe.execute()
                return finished_job
                break
            except ConnectionError:
                if attempt < MAX_RETRIES - 1:  # if not the last attempt
                    logging.error(f"Connection lost with Redis. Retrying...")
                    continue
                else:
                    logging.error(f"Connection lost with Redis. Max retries exceeded.")
                    raise
    elif job["status"] == "failed":
        if job["where"] == "img2img":
            raise HTTPException(status_code=500, status="Failed to do img2img")
        elif job["where"] == "txt2img":
            raise HTTPException(status_code=500, status="Failed to do txt2img")
        else:
            raise HTTPException(status_code=500, status="Unknown error retrieving job")
    else:
        # Calculate queue position
        queue_position = 1
        for j_id, j in jobs.items():
            if j["status"] == "running" or j["status"] == "processing":
                if j_id == job_id:
                    break
                queue_position += 1

        return JSONResponse({"status": job["status"], "queue_position": queue_position})


class JobRetryInfo(BaseModel):
    job_id: str
    indexes: List[int]


@app.get("/resend_images/{job_id}")
async def resend_images(JobRetryInfo: JobRetryInfo):
    job = jobs.get(JobRetryInfo.job_id)
    if job is None:
        raise HTTPException(status_code=500, status="Unknown error retrieving job")

    pipe = r.pipeline()
    for i, image in enumerate(job["result"]):
        if i in JobRetryInfo.indexes:
            byte_arr = io.BytesIO()
            # image.save(byte_arr, format='PNG')
            image.save(byte_arr, format="webp", quality=95)
            image_data = byte_arr.getvalue()
            pipe.set(f"job:{JobRetryInfo.job_id}:image:{i}", image_data)

            # Compute and store the checksum
            checksum = hashlib.sha256(image_data).hexdigest()
            pipe.set(f"job:{JobRetryInfo.job_id}:image:{i}:checksum", checksum)
    pipe.execute()
    return JSONResponse({"status": job["status"]})


def process_pending_jobs():
    while True:
        try:
            # Delete old jobs
            delete_old_jobs()
        except Exception as e:
            raise e
            print(e)

        # Filter jobs that are running
        job_list = [job for job in list(jobs.items()) if job[1]["status"] == "running"]

        # Sort jobs so that fast pass jobs are processed first
        job_list.sort(key=lambda x: x[1]["fast_pass_enabled"], reverse=True)

        for job_id, job in job_list:
            if job_id not in jobs:  # Job has been deleted
                continue

            try:
                t1 = time.time()
                process_image_task(
                    jobs[job_id]["request_data"],
                    job_id,
                    jobs[job_id]["request_data"].job_type,
                )
                deltatime = time.time() - t1
                print(f"Job {job_id} took {deltatime} seconds")
                break
            except Exception as e:
                raise e
                print(e)
                del jobs[job_id]

                # Write details of job into log file
                with open("log.txt", "a") as f:
                    f.write(f"Job {job_id} failed at {datetime.now()}\n")
                    f.write(f"Job details: {job}\n")
                    f.write(f"Error: {e}\n\n")


def start_job_processing_thread():
    job_processing_thread = threading.Thread(target=process_pending_jobs, daemon=True)
    job_processing_thread.start()


def delete_old_jobs():
    current_time = time.time()
    # Copy keys to a list to avoid RuntimeError: dictionary changed size during iteration
    for job_id in list(jobs.keys()):
        if (
            jobs[job_id]["status"] == "completed"
            and current_time - jobs[job_id]["timestamp"] > 60 * 60
        ):  # 60 minutes
            del jobs[job_id]


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert (
        image.shape[0:1] == image_mask.shape[0:1]
    ), "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def validate_weight(weight):
    try:
        float_weight = float(weight)
        if 0 <= float_weight <= 10:
            return str(float_weight)
        else:
            return "1.0"
    except ValueError:
        return "1.0"


def clean_tags(input_str):
    def replace_tag(match):
        tag, weight = match.groups()
        weight = validate_weight(weight)
        return f"({tag}:{weight})"

    return re.sub(r"\(([^:()]+):([^:()]+)\)", replace_tag, input_str)


start_job_processing_thread()
