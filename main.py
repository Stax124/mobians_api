import io
import base64
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from io import BytesIO
from uuid import uuid4
import threading
from multiprocessing import Pool
import asyncio

from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import tomesd
from PIL import Image, ImageDraw, ImageFont
import asyncio


class ImageRequestModel(BaseModel):
    data: dict
    model: str
    save_image: bool
    job_type: str

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
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

pipe.enable_vae_slicing()
pipe.enable_xformers_memory_efficient_attention()
pipe.load_textual_inversion("EasyNegativeV2.safetensors")
tomesd.apply_patch(pipe, ratio=0.3)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

def add_watermark(image):
    # Create watermark image
    watermark_text = "Mobians.ai"
    opacity = 128
    watermark = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(watermark)
    font_file_path = r'Roboto-Medium.ttf'
    font = ImageFont.truetype(font_file_path, 25)
    draw.text((10, 10), watermark_text, font=font, fill=(255, 255, 255, opacity))

    # Overlay watermark on the original image
    image_with_watermark = Image.alpha_composite(image.convert("RGBA"), watermark)
    return image_with_watermark

def process_generated_images(images):
    base64_images = []
    for image in images:
        img_io = io.BytesIO()

        image_with_watermark = add_watermark(image)

        image_with_watermark.save(img_io, format='PNG')
        #image_with_watermark.save(img_io, format='WEBP', quality=95)
        img_io.seek(0)
        base64_images.append(base64.b64encode(img_io.getvalue()).decode('utf-8'))
    return base64_images

def promptFilter(data):
    prompt = data.data['prompt']
    negative_prompt = data.data['negative_prompt']

    character_list = ['cream the rabbit', 
                      'rosy the rascal',
                      'sage',
                      'maria robotnik',
                      'marine the raccoon',
                      ]
    
    censored_tags = ['breast',
                     'nipples',
                     'pussy',
                     'nsfw',
                     'nudity',
                     'naked',
                     'loli',
                     'nude',
                     'ass',
                     'rape',
                     'sex',
                     'boob',
                     'sexy',
                     'busty',
                     'tits',
                     'thighs',
                     'thick',
                     'underwear',
                     'panties',
                     'upskirt',
                     'cum'
                     ]

    # If character is in prompt, filter out censored tags from prompt
    if any(character in prompt.lower() for character in character_list):
        for tag in censored_tags:
            prompt = prompt.replace(tag, '')
        negative_prompt = "sexy, nipples, pussy, breasts, " + negative_prompt
            
    return prompt, negative_prompt

def fortify_default_negative(negative_prompt):
    if "nsfw" in negative_prompt.lower() and "nipples" not in negative_prompt.lower():
        return "nipples, pussy, breasts, " + negative_prompt
    else:
        return negative_prompt

def process_image_task(request_data, job_id, job_type):
    request_data.data['prompt'], request_data.data['negative_prompt'] = promptFilter(request_data)
    request_data.data['negative_prompt'] = fortify_default_negative(request_data.data['negative_prompt'])
    request_data.data['seed'] = torch.Generator(device="cuda").manual_seed(request_data.data['seed'])

    if job_type == "txt2img":
        images = pipe.text2img(prompt=request_data.data['prompt'], 
                    negative_prompt=request_data.data['negative_prompt'], 
                    num_images_per_prompt=4, 
                    num_inference_steps=20, 
                    width=request_data.data['width'], 
                    height=request_data.data['height'],
                    guidance_scale=float(request_data.data['guidance_scale']),
                    generator=request_data.data['seed']
                    ).images
    elif job_type == "img2img":
        init_image = Image.open(BytesIO(base64.b64decode(request_data.data['image'].split(",", 1)[0]))).convert("RGB")
        tempAspectRatio = init_image.width / init_image.height
        if tempAspectRatio < 0.8:
            init_image = init_image.resize((512, 768))
        elif tempAspectRatio > 1.2:
            init_image = init_image.resize((768, 512))
        else:
            init_image = init_image.resize((512, 512))

        images = pipe.img2img(prompt=request_data.data['prompt'], 
                negative_prompt=request_data.data['negative_prompt'], 
                image=init_image,
                strength=float(request_data.data['strength']),
                num_images_per_prompt=4, 
                num_inference_steps=20, 
                guidance_scale=float(request_data.data['guidance_scale']),
                generator=request_data.data['seed']
                ).images
    else:
        print("Invalid job type")
        images = []
        
    base64_images = process_generated_images(images)
    jobs[job_id] = {'status': 'completed', 'result': base64_images}

@app.get("/get_queue_length/")
async def get_queue_length():
    # Return queue length where jobs are either running or processing
    queue_length = 0
    for j_id, j in jobs.items():
        if j['status'] == "running" or j['status'] == "processing":
            queue_length += 1
    return {"queue_length": queue_length}

@app.post("/generate_image/")
async def submit_job(request: ImageRequestModel):
    job_id = str(uuid4())
    jobs[job_id] = {"status": "running", 'request_data': request}
    return {"job_id": job_id}

@app.get("/get_job/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return {"status": "not found"}

    if job["status"] == "completed":
        finished_job = {"status": job['status'], 'result': job['result']}
        del jobs[job_id]
        return finished_job
    else:
        # Calculate queue position
        queue_position = 0
        for j_id, j in jobs.items():
            if j['status'] == "running" or j['status'] == "processing":
                if j_id == job_id:
                    break
                queue_position += 1

        return JSONResponse({"status": job['status'], "queue_position": queue_position})

def process_pending_jobs():
    while True:
        try:
            # Filter jobs that are running
            job_list = [job for job in jobs.items() if job[1]["status"] == "running"]
            
            # Sort jobs so that admin jobs are processed first
            job_list.sort(key=lambda x: 'admin' not in x[1]['request_data'].data['negative_prompt'])
            
            for job_id, job in job_list:
                if job_id not in jobs:  # Job has been deleted
                    continue
                process_image_task(jobs[job_id]['request_data'], job_id, jobs[job_id]['request_data'].job_type)
                break
        except Exception as e:
            print(e)

def start_job_processing_thread():
    job_processing_thread = threading.Thread(target=process_pending_jobs, daemon=True)
    job_processing_thread.start()


# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(process_pending_jobs())


start_job_processing_thread()
