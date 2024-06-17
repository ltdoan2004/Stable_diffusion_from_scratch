import torch
import torch.nn 
from torch.nn import functional as F
import tqdm
from ddpm import DDPMSampler
import numpy as np

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH = WIDTH // 8
LATENT_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not( 0 < strength < 1):
            raise ValueError("strenth must be 0 to 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            #The tensor values are generated randomly, 
            #but because the seed is fixed, the "random" values are the same every time you run the script.
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype= torch.long, device =device)
            cond_embed = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding = "max_length", max_length = 77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype = torch.long, device = device)
            uncond_embed = clip(uncond_tokens)

            context = torch.cat([cond_embed, uncond_embed])
        else: 
            tokens = tokenizer.batch_encode_plus([prompt], padding = "max_length", max_length = 77).input_ids
            tokens = torch.tensor(cond_tokens, dtype= torch.long, device =device)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else: 
            raise ValueError("Unknown sampler") 
        
        latent_shape = (1 ,4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            #h , w , c
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor, dtype = torch.float32)
            input_image_tensor = torch.tensor(input_image_tensor, dtype = torch.float32, device = device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            #h , w , c -> bs, h , w , c
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            encoder_noise = torch.randn(latent_shape, generator = generator, device =device)

            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength = strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.rand(latent_shape, generator = generator, device = device)
        
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2,1,1,1)

            #model output is the predicted noise by the UNET model
            model_output = diffusion(latents, context, time_embedding)

            if do_cfg:
                output_cond , output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1,1), (0,255), clamp = True)
        images = images.permute(0,2,3,1)
        images = images.to("cpu", torch.uint8).numpy()


def rescale(x, old_rangle, new_range, clamp = False):
    old_min , old_max = old_rangle
    new_min , new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += old_max
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start =0, end =160, dtype = torch.float32) / 160)
    x = torch.tensor([timestep], dtype = torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)])

