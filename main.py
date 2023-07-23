import io
import base64
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from io import BytesIO
from uuid import uuid4
import threading
from datetime import datetime
from typing import Optional, List
import time
import hashlib
import logging

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from diffusers.utils import load_image, randn_tensor
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler, StableDiffusionPipeline
import torch
import numpy as np
import tomesd
from PIL import Image, ImageDraw, ImageFont
import PIL.ImageOps
import asyncio
import redis
from redis.backoff import ExponentialBackoff
from redis.retry import Retry
from redis.exceptions import (
   BusyLoadingError,
   ConnectionError,
   TimeoutError
)

# Run 3 retries with exponential backoff strategy
retry = Retry(ExponentialBackoff(), 3)

r = redis.Redis(host='76.157.184.213', port=6379, db=0, retry=retry, retry_on_error=[BusyLoadingError, ConnectionError, TimeoutError])

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

pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path="testSonicBeta4", 
                                        custom_pipeline="lpw_stable_diffusion",
                                        torch_dtype=torch.float16, 
                                        revision="fp16",
                                        safety_checker=None,
                                        feature_extractor=None,
                                        requires_safety_checker=False,).to("cuda")
#text2img.unet = torch.compile(text2img.unet, mode="reduce-overhead", fullgraph=True)

pipe.enable_vae_slicing()
# pipe.enable_xformers_memory_efficient_attention()
pipe.load_textual_inversion("EasyNegativeV2.safetensors")
tomesd.apply_patch(pipe, ratio=0.3)


pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# pipe.load_lora_weights('more_details.safetensors')

# Controlnet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", 
    torch_dtype=torch.float16
).to("cuda")
inpainting = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "testSonicBeta4", controlnet=controlnet, 
    torch_dtype=torch.float16
).to("cuda")
inpainting.scheduler = EulerAncestralDiscreteScheduler.from_config(inpainting.scheduler.config)
# inpainting.scheduler = UniPCMultistepScheduler.from_config(inpainting.scheduler.config)
#inpainting.scheduler = DDIMScheduler.from_config(inpainting.scheduler.config)
inpainting.enable_vae_slicing()
tomesd.apply_patch(inpainting, ratio=0.3)

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


def process_image_task(request_data, job_id, job_type):
    # Seed filtering is done on django so it doesnt crash the program (hopefully)
    seed = torch.Generator(device="cuda").manual_seed(request_data.seed)

    if job_type == "txt2img":
        try:
            images = pipe.text2img(prompt=request_data.prompt, 
                        negative_prompt=request_data.negative_prompt, 
                        num_images_per_prompt=4, 
                        num_inference_steps=20, 
                        width=request_data.width, 
                        height=request_data.height,
                        guidance_scale=float(request_data.guidance_scale),
                        generator=seed,
                        # cross_attention_kwargs={"scale": 1}
                        ).images
        except Exception as e:
            print(e)
            time_stamp = jobs[job_id]['timestamp']
            jobs[job_id] = {'status': 'failed', 'where':'txt2img', 'timestamp': time_stamp}

    elif job_type == "img2img":
        try:
            # Convert base64 string to PIL Image
            base64_image = request_data.image
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))

            images = pipe.img2img(prompt=request_data.prompt, 
                    negative_prompt=request_data.negative_prompt, 
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
            time_stamp = jobs[job_id]['timestamp']
            jobs[job_id] = {'status': 'failed', 'where':'img2img', 'timestamp': time_stamp}
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

            control_image = make_inpaint_condition(image, mask_image)
            # control_image.show()

            images = inpainting(prompt=request_data.prompt, 
                    negative_prompt=request_data.negative_prompt, 
                    image=image,
                    mask_image=mask_image,
                    strength=float(request_data.strength),
                    num_images_per_prompt=4, 
                    num_inference_steps=20, 
                    guidance_scale=float(request_data.guidance_scale),
                    generator=seed,
                    control_image=control_image
                    ).images
        except Exception as e:
            print(e)
            time_stamp = jobs[job_id]['timestamp']
            jobs[job_id] = {'status': 'failed', 'where':'inpainting', 'timestamp': time_stamp}  
    elif job_type == "tile":
        try:
            # Convert base64 string to PIL Image
            base64_image = request_data.image
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))

            images = tile(prompt=request_data.prompt, 
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
            time_stamp = jobs[job_id]['timestamp']
            jobs[job_id] = {'status': 'failed', 'where':'tile controlnet', 'timestamp': time_stamp}
    else:
        print("Invalid job type")
        return
        
    # Convert PIL Image objects to base64 strings
    #images = [image_to_base64(img) for img in images]
    time_stamp = jobs[job_id]['timestamp']
    jobs[job_id] = {'status': 'completed', 'result': images, 'timestamp': time_stamp, 'request_data': request_data}

@app.get("/get_queue_length/")
async def get_queue_length():
    # Return queue length where jobs are either running or processing
    queue_length = 0
    for j_id, j in jobs.items():
        if j['status'] == "running" or j['status'] == "processing":
            queue_length += 1
    # return {"queue_length": queue_length}
    return JSONResponse({"queue_length": queue_length})


@app.post("/submit_job/")
async def submit_job(request: ImageRequestModel):
    job_id = str(uuid4())
    jobs[job_id] = {"status": "running", 'request_data': request, 'timestamp': time.time(), 'fast_pass_enabled': request.fast_pass_enabled}
    return JSONResponse({"job_id": job_id})

@app.get("/get_job/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "completed":
        finished_job = {"status": job['status']}

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                pipe = r.pipeline()
                for i, image in enumerate(job['result']):
                    byte_arr = io.BytesIO()
                    # image.save(byte_arr, format='PNG')
                    image.save(byte_arr, format='webp', quality=95)
                    image_data = byte_arr.getvalue()
                    pipe.set(f"job:{job_id}:image:{i}", image_data)

                    # Compute and store the checksum
                    checksum = hashlib.sha256(image_data).hexdigest() 
                    pipe.set(f"job:{job_id}:image:{i}:checksum", checksum)

                    # Include metadata
                    metadata = job['request_data'].json()
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
            if j['status'] == "running" or j['status'] == "processing":
                if j_id == job_id:
                    break
                queue_position += 1

        return JSONResponse({"status": job['status'], "queue_position": queue_position})
    

class JobRetryInfo(BaseModel):
    job_id: str
    indexes: List[int]

@app.get("/resend_images/{job_id}")
async def resend_images(JobRetryInfo: JobRetryInfo):
    job = jobs.get(JobRetryInfo.job_id)
    if job is None:
        raise HTTPException(status_code=500, status="Unknown error retrieving job")
    
    pipe = r.pipeline()
    for i, image in enumerate(job['result']):
        if i in JobRetryInfo.indexes:
            byte_arr = io.BytesIO()
            # image.save(byte_arr, format='PNG')
            image.save(byte_arr, format='webp', quality=95)
            image_data = byte_arr.getvalue()
            pipe.set(f"job:{JobRetryInfo.job_id}:image:{i}", image_data)

            # Compute and store the checksum
            checksum = hashlib.sha256(image_data).hexdigest()
            pipe.set(f"job:{JobRetryInfo.job_id}:image:{i}:checksum", checksum)
    pipe.execute()
    return JSONResponse({"status": job['status']})

def process_pending_jobs():
    while True:
        try:
            # Delete old jobs
            delete_old_jobs()
        except Exception as e:
            print(e)
            
        # Filter jobs that are running
        job_list = [job for job in list(jobs.items()) if job[1]["status"] == "running"]

        # Sort jobs so that fast pass jobs are processed first
        job_list.sort(key=lambda x: x[1]['fast_pass_enabled'], reverse=True)
        
        for job_id, job in job_list:
            if job_id not in jobs:  # Job has been deleted
                continue

            try:
                process_image_task(jobs[job_id]['request_data'], job_id, jobs[job_id]['request_data'].job_type)
                break
            except Exception as e:
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
        if jobs[job_id]['status'] == 'completed' and current_time - jobs[job_id]['timestamp'] > 20 * 60:  # 20 minutes
            del jobs[job_id]

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
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

start_job_processing_thread()
