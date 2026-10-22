import gc

import streamlit as st
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _unload_current_model():
    if "model" in st.session_state and st.session_state["model"] is not None:
        del st.session_state["model"]
        del st.session_state["tokenizer"]
        st.session_state["model"] = None
        st.session_state["tokenizer"] = None
        st.session_state["loaded_model_name"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _ensure_400M_model():
    model_name = "NovaSearch/stella_en_400M_v5"
    if st.session_state.get("loaded_model_name") == model_name:
        return
    _unload_current_model()
    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.use_memory_efficient_attention = False
    config.unpad_inputs = False

    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.half()
    model = model.to(device)
    model.eval()

    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer
    st.session_state["loaded_model_name"] = model_name
    st.session_state["device"] = device


def _ensure_1_5B_model():
    model_name = "NovaSearch/stella_en_1.5B_v5"
    if st.session_state.get("loaded_model_name") == model_name:
        return
    _unload_current_model()
    device = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Stella 1.5B Qwen Patching
    import transformers.utils as _tu
    for attr, fn in [("is_flash_attn_greater_or_equal_2_10", lambda: False),
                     ("is_flash_attn_2_available",           lambda: False),
                     ("is_flash_attn_greater_or_equal",      lambda v: False)]:
        if not hasattr(_tu, attr): setattr(_tu, attr, fn)

    from transformers import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, n, p=0: p

    from transformers import Qwen2Config
    if not hasattr(Qwen2Config, "_rope_theta_patched"):
        _orig_getattr = Qwen2Config.__getattribute__
        def _qwen2_getattr(self, name):
            if name == "rope_theta":
                try: return _orig_getattr(self, name)
                except AttributeError: return 1000000.0
            return _orig_getattr(self, name)
        Qwen2Config.__getattribute__ = _qwen2_getattr
        Qwen2Config._rope_theta_patched = True

    model = AutoModel.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer
    st.session_state["loaded_model_name"] = model_name
    st.session_state["device"] = device
