"""Load the models used in the project."""
import gc
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV_MODEL = "Qwen/Qwen2.5-3B-Instruct"    # fast, full 16-bit precision: write and test code here
MAIN_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # final results, 4-bit so it fits on a T4


def load_model(name=DEV_MODEL, four_bit=None):
    """Return (model, tokenizer) in eval mode.

    four_bit=None picks 4-bit automatically for 7B/8B models (they do not fit a T4 otherwise).
    """
    if four_bit is None:
        four_bit = any(s in name for s in ("7B", "8B"))
    tok = AutoTokenizer.from_pretrained(name)
    cuda = torch.cuda.is_available()
    kw = dict(device_map="cuda" if cuda else "cpu", low_cpu_mem_usage=True)
    if four_bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["dtype"] = torch.float16 if cuda else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except TypeError:  # older transformers: the argument was called torch_dtype
        if "dtype" in kw:
            kw["torch_dtype"] = kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return model, tok


def model_info(model):
    """Number of layers, heads and sizes, read from the config."""
    c = model.config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return dict(n_layers=c.num_hidden_layers, n_heads=c.num_attention_heads,
                n_kv_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
                d_model=c.hidden_size, head_dim=head_dim)


def free_model(model=None, name=None):
    """Release GPU memory and (optionally) delete the downloaded weights from the Colab disk."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if name:
        path = os.path.expanduser("~/.cache/huggingface/hub/models--" + name.replace("/", "--"))
        shutil.rmtree(path, ignore_errors=True)
