from typing import Union
import io
import json
import base64
from pydantic import BaseModel
from typing import Union
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
import torch
import tomesd
from PIL import Image, ImageDraw, ImageFont



class ImageRequestModel(BaseModel):
    data: dict
    model: str
    save_image: bool

app = FastAPI()

# Set up the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path="testSonicBeta4", 
                                        torch_dtype=torch.float16, 
                                        safety_checker=None,
                                        feature_extractor=None,
                                        requires_safety_checker=False,).to("cuda")
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
#pipe.enable_xformers_memory_efficient_attention()
tomesd.apply_patch(pipe, ratio=0.3)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path="testSonicBeta4", 
                                        torch_dtype=torch.float16, 
                                        safety_checker=None,
                                        feature_extractor=None,
                                        requires_safety_checker=False,).to("cuda")
pipe_img2img.enable_model_cpu_offload()
#pipe_img2img.enable_xformers_memory_efficient_attention()
pipe_img2img.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_img2img.scheduler.config)
#pipe_inpaint.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
tomesd.apply_patch(pipe_img2img, ratio=0.3)


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

        #image.save(img_io, format='PNG')
        image_with_watermark.save(img_io, format='WEBP', quality=95)
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
    

@app.post("/txt2img")
async def txt2img(request_data: ImageRequestModel):
    request_data.data['prompt'], request_data.data['negative_prompt'] = promptFilter(request_data)
    request_data.data['negative_prompt'] = fortify_default_negative(request_data.data['negative_prompt'])
    request_data.data['seed'] = torch.Generator(device="cuda").manual_seed(request_data.data['seed'])

    images = pipe(prompt=request_data.data['prompt'], 
                negative_prompt=request_data.data['negative_prompt'], 
                num_images_per_prompt=4, 
                num_inference_steps=20, 
                width=request_data.data['width'], 
                height=request_data.data['height'],
                guidance_scale=float(request_data.data['guidance_scale']),
                generator=request_data.data['seed']).images

    base64_images = process_generated_images(images)
    
    return JSONResponse(content={'images': base64_images})

@app.post("/img2img")
async def img2img(request_data: ImageRequestModel):
    request_data.data['prompt'], request_data.data['negative_prompt'] = promptFilter(request_data)
    request_data.data['negative_prompt'] = fortify_default_negative(request_data.data['negative_prompt'])
    request_data.data['seed'] = torch.Generator(device="cuda").manual_seed(request_data.data['seed'])

    init_image = Image.open(BytesIO(base64.b64decode(request_data.data['image'].split(",", 1)[0]))).convert("RGB")
    tempAspectRatio = init_image.width / init_image.height
    if tempAspectRatio < 0.8:
        init_image = init_image.resize((512, 768))
    elif tempAspectRatio > 1.2:
        init_image = init_image.resize((768, 512))
    else:
        init_image = init_image.resize((512, 512))

    images = pipe_img2img(prompt=request_data.data['prompt'], 
                negative_prompt=request_data.data['negative_prompt'], 
                image=init_image,
                strength=float(request_data.data['strength']),
                num_images_per_prompt=4, 
                num_inference_steps=20, 
                guidance_scale=float(request_data.data['guidance_scale']),
                generator=request_data.data['seed']
                ).images

    base64_images = process_generated_images(images)
    
    return JSONResponse(content={'images': base64_images})
