"""Offline LTX-Video 2B (v0.9.1) text-to-video and image-to-video.

  python ltx_gen.py --mode t2v --prompt "a candle flickering" --name test
  python ltx_gen.py --mode i2v --image C:\\path\\pic.png --prompt "sparkles drift upward" --name test2

Videos are saved to the "outputs" folder next to this file.
"""
import argparse, gc, os, sys, time

import ltx_common as C  # folders, model paths and the optional extra package folder

ROOT = C.HOME

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer, T5Config, T5EncoderModel
from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXImageToVideoPipeline,
                       LTXPipeline, LTXVideoTransformer3DModel)
from diffusers.utils import export_to_video

CKPT = C.CKPT
T5_FILE = C.T5_FILE
T5_CFG = C.T5_CFG
TOK = C.TOKENIZER
CFG = C.CFG_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["t2v", "i2v"], required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--negative", default="worst quality, inconsistent motion, blurry, jittery, distorted")
ap.add_argument("--image", help="start image for i2v")
ap.add_argument("--name", default="video")
ap.add_argument("--w", type=int, default=512)
ap.add_argument("--h", type=int, default=320)
ap.add_argument("--frames", type=int, default=25, help="must be 8*k+1")
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--fps", type=int, default=24)
a = ap.parse_args()
assert a.w % 32 == 0 and a.h % 32 == 0 and (a.frames - 1) % 8 == 0, "w/h need to be multiples of 32 and frames = 8*k+1"
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


# ---- stage 1: encode the prompts with T5 (fp8 weights on the GPU, upcast one layer at a time)
log("loading text encoder")
tok = AutoTokenizer.from_pretrained(TOK, use_fast=False)
with torch.device("meta"):
    te = T5EncoderModel(T5Config.from_pretrained(T5_CFG))
sd = load_file(T5_FILE, device="cuda")
res = te.load_state_dict(sd, strict=False, assign=True)
te.encoder.embed_tokens = te.shared
left = [n for n, p in te.named_parameters() if p.device.type == "meta"]
print("missing keys:", [k for k in res.missing_keys if "embed_tokens" not in k], "| still on meta:", left)
del sd


def upcast(m, _args):
    m._fp8 = {n: p.data for n, p in m.named_parameters()}
    for n, p in m.named_parameters():
        p.data = p.data.to(torch.bfloat16)


def restore(m, _args, _out):
    for n, p in m.named_parameters():
        p.data = m._fp8[n]


# one block (~400 MB in bf16) is upcast at a time; embeddings and the final norm are handled the same way
for m in [te.shared, te.encoder.final_layer_norm, *te.encoder.block]:
    m.register_forward_pre_hook(upcast)
    m.register_forward_hook(restore)
te.eval()


@torch.no_grad()
def encode(text):
    t = tok(text, padding="max_length", max_length=128, truncation=True, add_special_tokens=True, return_tensors="pt")
    emb = te(t.input_ids.to("cuda"))[0]
    return emb.to(torch.bfloat16).cpu(), t.attention_mask.bool().cpu()


pe, pm = encode(a.prompt)
ne, nm = encode(a.negative)
log(f"prompt encoded {tuple(pe.shape)}")
del te
gc.collect(); torch.cuda.empty_cache()

# ---- stage 2: the video model, streamed tensor-by-tensor to the GPU (keeps system RAM use low)
log("loading video model")
import json
from accelerate import init_empty_weights
from diffusers.loaders.single_file_utils import (convert_ltx_transformer_checkpoint_to_diffusers,
                                                 convert_ltx_vae_checkpoint_to_diffusers)
from safetensors import safe_open

raw = {}
with safe_open(CKPT, framework="pt", device="cpu") as f:
    for k in f.keys():
        raw[k] = f.get_tensor(k).to("cuda", torch.bfloat16)
tr_sd = convert_ltx_transformer_checkpoint_to_diffusers(raw)
vae_sd = convert_ltx_vae_checkpoint_to_diffusers(raw)
del raw


def build(cls, sub, sd):
    cfg = json.load(open(os.path.join(CFG, sub, "config.json")))
    with init_empty_weights():
        model = cls.from_config(cfg)
    r = model.load_state_dict(sd, strict=False, assign=True)
    print(sub, "missing:", r.missing_keys[:6], "unexpected:", r.unexpected_keys[:6])
    return model.to("cuda").eval()


transformer = build(LTXVideoTransformer3DModel, "transformer", tr_sd)
vae = build(AutoencoderKLLTXVideo, "vae", vae_sd)
del tr_sd, vae_sd
gc.collect(); torch.cuda.empty_cache()
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(CFG, subfolder="scheduler")
cls = LTXPipeline if a.mode == "t2v" else LTXImageToVideoPipeline
pipe = cls(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)
pipe.vae.enable_tiling()

kw = dict(prompt_embeds=pe.to("cuda"), prompt_attention_mask=pm.to("cuda"), negative_prompt_embeds=ne.to("cuda"),
          negative_prompt_attention_mask=nm.to("cuda"),
          width=a.w, height=a.h, num_frames=a.frames, num_inference_steps=a.steps, guidance_scale=a.cfg,
          generator=torch.Generator("cuda").manual_seed(a.seed))
if a.mode == "i2v":
    img = Image.open(a.image)
    if img.mode == "RGBA":  # transparent background -> white so the model gets a normal picture
        bg = Image.new("RGB", img.size, (255, 255, 255)); bg.paste(img, mask=img.getchannel("A")); img = bg
    kw["image"] = img.convert("RGB").resize((a.w, a.h), Image.LANCZOS)
log("generating")
frames = pipe(**kw).frames[0]
out = os.path.join(ROOT, "outputs", f"{a.name}.mp4")
os.makedirs(os.path.dirname(out), exist_ok=True)
export_to_video(frames, out, fps=a.fps)
log(f"saved {out}")
