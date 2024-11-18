from typing import Optional, Union, List
from functools import lru_cache
import torch
from diffusers import AudioLDM2Pipeline, StableAudioPipeline, DPMSolverMultistepScheduler


def get_audioldm2_pipe(pretrained_model_name_or_path: Optional[str] = None,
                       device: str = 'cpu') -> AudioLDM2Pipeline:
    pipe = AudioLDM2Pipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe


def get_stableaudio_pipe(pretrained_model_name_or_path: Optional[str] = None,
                         device: str = 'cpu') -> StableAudioPipeline:
    pipe = StableAudioPipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.float16
    )
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention(attention_op=None)
    pipe.enable_model_cpu_offload()
    return pipe


@lru_cache(maxsize=1)
def get_pipe(pretrained_model_name_or_path: Optional[str] = None,
             device: str = 'cpu') -> Union[AudioLDM2Pipeline, StableAudioPipeline]:
    if 'ldm' in pretrained_model_name_or_path:
        pipe = get_audioldm2_pipe(pretrained_model_name_or_path, device)
    elif 'stable' in pretrained_model_name_or_path:
        pipe = get_stableaudio_pipe(pretrained_model_name_or_path, device)
    else:
        raise ValueError(f'Model: {pretrained_model_name_or_path} not valid.')
    return pipe


def set_torch_seed(seed: int, device: str) -> torch.Generator:
    generator = torch.Generator(device=device).manual_seed(seed)
    return generator


def run_pipe(pipe: Union[AudioLDM2Pipeline, StableAudioPipeline],
             prompt: Union[List[str], str],
             negative_prompt: str,
             audio_length_in_s: float,
             random_generator: torch.Generator,
             guidance_scale: float,
             num_inference_steps: int,
             num_waveforms_per_prompt: int,
             initial_audio_sampling_rate: Optional[float] = None
             ) -> dict:
    if isinstance(pipe, AudioLDM2Pipeline):
        return pipe(prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    audio_length_in_s=audio_length_in_s,
                    num_waveforms_per_prompt=num_waveforms_per_prompt,
                    generator=random_generator,
                    output_type="pt").audios
    else:
        return pipe(prompt=prompt,
                    negative_prompt=negative_prompt,
                    audio_end_in_s=audio_length_in_s,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_waveforms_per_prompt=num_waveforms_per_prompt,
                    generator=random_generator,
                    initial_audio_sampling_rate=torch.tensor(initial_audio_sampling_rate, device=pipe.device)).audios
