from diffusers import StableDiffusionPipeline
import torch

#.\myenv\scripts\activate.ps1
# https://huggingface.co/runwayml/stable-diffusion-v1-5

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
#pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
#pipe = pipe.to("cpu")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
