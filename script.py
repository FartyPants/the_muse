
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)
import gradio as gr

from transformers.generation import LogitsWarper
import torch

params = {
    "display_name": "The Muse",
    "is_tab": False,
    "enable": True,
    "temperature": 0.7,
    "top_p":0.7,
    "top_k":50,
    "damp":0.8,
    "top_k_m": 3,
    "damp_initial": 1.0,
    "damp_ramp_tokens": 25,

}

class MuseLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs dampening of the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        damp (`float`, *optional*, defaults to 0.98):
            How much less likely should the top_k most likely tokens be made. If set to 0, they become impossible.
    """

    def __init__(self, top_k: int, damp: float = 0.98, damp_initial: float = 1.0, damp_ramp_tokens: int = 0, min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.damp = damp
        self.damp_initial = damp_initial
        self.damp_ramp_tokens = damp_ramp_tokens
        self.token_num = 0

    def reset(self):
        self.token_num = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check

        ratio = 1.0 if self.damp_ramp_tokens == 0 else min(self.token_num/self.damp_ramp_tokens, 1.0)        
        linear_damp = self.damp_initial + ratio*(self.damp - self.damp_initial) if ratio < 1.0 else self.damp

        topk_values, topk_indices = torch.topk(scores, top_k)
        dampened_values = topk_values * linear_damp
        scores.scatter_(-1, topk_indices, dampened_values)

        self.token_num += 1

        return scores


def logits_processor_modifier(processor_list, input_ids):
    """
    Adds logits processors to the list, allowing you to access and modify
    the next token probabilities.
    Only used by loaders that use the transformers library for sampling.
    """
    if params["enable"]:
        processor_list.clear()
        print("Muse loggits ON")
        from transformers.generation import TopPLogitsWarper, TopKLogitsWarper, TemperatureLogitsWarper
        processor_list.append(TemperatureLogitsWarper(temperature=params["temperature"]))
        the_muse = MuseLogitsWarper(top_k=params["top_k_m"], damp=params["damp"], damp_initial =params["damp_initial"] , damp_ramp_tokens=params["damp_ramp_tokens"] )
        processor_list.append(the_muse)
        processor_list.append(TopPLogitsWarper(top_p=params["top_p"]))
        processor_list.append(TopKLogitsWarper(top_k=params["top_k"]))
    return processor_list


def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.
    
    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """

    enable = gr.Checkbox(value=True,label="Enable The Muse")
    enable.change(lambda x: params.update({'enable': x}), enable, None)
    with gr.Accordion("Base Parameters", open=False):
        temperature = gr.Slider(0.01, 1.99, value=params["temperature"], step=0.01, label='temperature')
        top_p = gr.Slider(0.0, 1.0, value=params["top_p"], step=0.01, label='top_p')
        top_k = gr.Slider(0, 200, value=params["top_k"], step=1, label='top_k')
    with gr.Accordion("Probabilities reduction"):
        top_k_m = gr.Slider(1, 40, value=params["top_k_m"], step=1, label='Reduce the Most probable tokens [top_k]')
        #damp_initial = gr.Slider(0.0, 1.0, value=params["damp_initial"], step=0.01, label='- from [damp_initial]')
        damp = gr.Slider(0.0, 1.0, value=params["damp"], step=0.01, label='with probability [damp]')
        damp_ramp_tokens = gr.Slider(0, 100, value=params["damp_ramp_tokens"], step=1, label='over the span of the first number of tokens [damp_ramp_tokens]')
   
    temperature.change(lambda x: params.update({'temperature': x}), temperature, None)
    top_p.change(lambda x: params.update({'top_p': x}), top_p, None)
    top_k.change(lambda x: params.update({'top_k': x}), top_k, None)
    top_k_m.change(lambda x: params.update({'top_k_m': x}), top_k_m, None)
    #damp_initial.change(lambda x: params.update({'damp_initial': x}), damp_initial, None)
    damp.change(lambda x: params.update({'damp': x}), damp, None)
    damp_ramp_tokens.change(lambda x: params.update({'damp_ramp_tokens': x}), damp_ramp_tokens, None)


   

