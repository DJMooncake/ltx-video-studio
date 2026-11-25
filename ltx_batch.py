"""Batch LTX-Video (offline): loads the models ONCE, then makes every shot listed in a job file.

Used by LTX Video Studio v3.   Usage:  python ltx_batch.py job.json
Lines that start with @@ are progress messages that the app reads.

Shot types in the job file:
  t2v   text to video
  i2v   starts from a keyframe picture ("image")
  cont  starts from the last frame of the previous shot (chaining)
"""
import gc
import json
import os
import sys
import time
import traceback

import ltx_common as C  # folders, model paths and the optional extra package folder

ROOT = C.HOME

import imageio.v2 as iio
import torch
from accelerate import init_empty_weights
from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXImageToVideoPipeline,
                       LTXPipeline, LTXVideoTransformer3DModel)
from diffusers.loaders.single_file_utils import (convert_ltx_transformer_checkpoint_to_diffusers,
                                                 convert_ltx_vae_checkpoint_to_diffusers)
from diffusers.utils import export_to_video
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer, T5Config, T5EncoderModel

CKPT = C.CKPT
T5_FILE = C.T5_FILE
T5_CFG = C.T5_CFG
TOK = C.TOKENIZER
CFG = C.CFG_DIR

job = json.load(open(sys.argv[1], encoding="utf-8"))
OUT_DIR = job["out_dir"]
SHOTS = job["shots"]
N = len(SHOTS)
W, H, FRAMES, STEPS = job["w"], job["h"], job["frames"], job["steps"]
CFG_SCALE, SEED, FPS = job["cfg"], job["seed"], job["fps"]
os.makedirs(OUT_DIR, exist_ok=True)
t_start = time.time()


def say(msg):
    print(msg, flush=True)


def shot_path(i):
    return os.path.join(OUT_DIR, f"shot_{i:03d}.mp4")


def is_done(i):
    p = shot_path(i)
    if not (os.path.isfile(p) and os.path.getsize(p) > 1000):
        return False
    if "skip" in job:  # the app decides, based on each shot's settings, which finished clips are still valid
        return i in set(job["skip"])
    return bool(job.get("resume"))


def archive_old(path):
    """Never overwrite a finished clip: move it into an 'old' folder first."""
    if os.path.isfile(path):
        old = os.path.join(os.path.dirname(path), "old")
        os.makedirs(old, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        os.replace(path, os.path.join(old, os.path.basename(path)[:-4] + f"_{stamp}.mp4"))


def last_frame(path):
    last = None
    for f in iio.get_reader(path):
        last = f
    return Image.fromarray(last)


def prep_picture(path, i):
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))  # transparent areas become white
    bg.paste(img, mask=img.getchannel("A"))
    a, t = bg.width / bg.height, W / H
    kept = min(a / t, t / a)
    if kept < 0.85:
        say(f"@@NOTE shot {i}: your picture is a different shape from the video size, so about "
            f"{round((1 - kept) * 100)}% of it was cropped. Pick a matching Size to avoid that.")
    s = max(W / bg.width, H / bg.height)
    bg = bg.resize((max(W, round(bg.width * s)), max(H, round(bg.height * s))), Image.LANCZOS)
    left, top = (bg.width - W) // 2, (bg.height - H) // 2
    return bg.crop((left, top, left + W, top + H))


todo = [i for i in range(1, N + 1) if not is_done(i)]
say(f"@@STAGE {len(todo)} of {N} shots to make ({N - len(todo)} already done).")
embeds = {}
neg_emb = None

# ---- stage 1: understand every prompt with T5 (fp8 weights on the GPU, one block upcast at a time)
if todo:
    say("@@STAGE Step 1/3: loading the text encoder (once for all shots)...")
    tok = AutoTokenizer.from_pretrained(TOK, use_fast=False)
    with torch.device("meta"):
        te = T5EncoderModel(T5Config.from_pretrained(T5_CFG))
    sd = load_file(T5_FILE, device="cuda")
    res = te.load_state_dict(sd, strict=False, assign=True)
    te.encoder.embed_tokens = te.shared
    left_meta = [n for n, p in te.named_parameters() if p.device.type == "meta"]
    if left_meta:
        say(f"@@NOTE text encoder: some weights are missing: {left_meta[:3]}")
    del sd

    def upcast(m, _args):
        m._fp8 = {n: p.data for n, p in m.named_parameters()}
        for n, p in m.named_parameters():
            p.data = p.data.to(torch.bfloat16)

    def restore(m, _args, _out):
        for n, p in m.named_parameters():
            p.data = m._fp8[n]

    for m in [te.shared, te.encoder.final_layer_norm, *te.encoder.block]:
        m.register_forward_pre_hook(upcast)
        m.register_forward_hook(restore)
    te.eval()

    @torch.no_grad()
    def encode(text):
        t = tok(text, padding="max_length", max_length=128, truncation=True, add_special_tokens=True,
                return_tensors="pt")
        emb = te(t.input_ids.to("cuda"))[0]
        return emb.to(torch.bfloat16).cpu(), t.attention_mask.bool().cpu()

    neg_emb = encode(job["negative"])
    for k, i in enumerate(todo, 1):
        embeds[i] = encode(SHOTS[i - 1]["prompt"])
    say(f"@@STAGE prompts understood ({len(embeds)} shots).")
    del te
    gc.collect()
    torch.cuda.empty_cache()

    # ---- stage 2: the video model, streamed to the GPU tensor by tensor (keeps system RAM use low)
    say("@@STAGE Step 2/3: loading the video model (once for all shots)...")
    raw = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k).to("cuda", torch.bfloat16)
    tr_sd = convert_ltx_transformer_checkpoint_to_diffusers(raw)
    vae_sd = convert_ltx_vae_checkpoint_to_diffusers(raw)
    del raw

    def build(cls, sub, state):
        cfg = json.load(open(os.path.join(CFG, sub, "config.json")))
        with init_empty_weights():
            model = cls.from_config(cfg)
        r = model.load_state_dict(state, strict=False, assign=True)
        if r.missing_keys or r.unexpected_keys:
            say(f"@@NOTE {sub}: missing {r.missing_keys[:4]} unexpected {r.unexpected_keys[:4]}")
        return model.to("cuda").eval()

    transformer = build(LTXVideoTransformer3DModel, "transformer", tr_sd)
    vae = build(AutoencoderKLLTXVideo, "vae", vae_sd)
    del tr_sd, vae_sd
    gc.collect()
    torch.cuda.empty_cache()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(CFG, subfolder="scheduler")
    vae.enable_tiling()
    pipe_t2v = LTXPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer)
    pipe_i2v = LTXImageToVideoPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None,
                                       transformer=transformer)
    say("@@STAGE Step 3/3: making the shots...")

# ---- stage 3: the shots, in order
prev_last = None
made = skipped = 0
for i in range(1, N + 1):
    s = SHOTS[i - 1]
    say(f"@@SHOT {i} {N}")
    path = shot_path(i)
    try:
        if is_done(i):
            skipped += 1
            if i < N and SHOTS[i]["type"] == "cont":
                prev_last = last_frame(path)  # needed so the next shot can continue from it
            say(f"@@SKIP {i} {N} {path}")
            continue
        t0 = time.time()
        start = None
        if s["type"] == "i2v":
            start = prep_picture(s["image"], i)
        elif s["type"] == "cont":
            if prev_last is None:
                raise RuntimeError("this shot continues from the previous one, but there is no previous shot")
            start = prev_last
        pe, pm = embeds[i]
        kw = dict(prompt_embeds=pe.to("cuda"), prompt_attention_mask=pm.to("cuda"),
                  negative_prompt_embeds=neg_emb[0].to("cuda"), negative_prompt_attention_mask=neg_emb[1].to("cuda"),
                  width=W, height=H, num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=CFG_SCALE,
                  generator=torch.Generator("cuda").manual_seed(SEED + i))
        if start is not None:
            kw["image"] = start
        pipe = pipe_i2v if start is not None else pipe_t2v
        frames = pipe(**kw).frames[0]
        if s["type"] == "cont":
            frames = frames[1:]  # the first frame repeats the previous shot's last frame
        tmp = os.path.join(OUT_DIR, f"shot_{i:03d}.tmp.mp4")
        export_to_video(frames, tmp, fps=FPS)
        archive_old(path)
        os.replace(tmp, path)  # only finished clips ever carry the real name, so resume is safe
        prev_last = frames[-1]
        made += 1
        say(f"@@DONE {i} {N} {path} {time.time() - t0:.1f}")
        del frames, kw
        gc.collect()
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        say(f"@@ERROR shot {i}: the graphics card ran out of memory. Close other programs or pick a smaller "
            f"Size / shorter length, then run again with the same Save as name to carry on from here.")
        sys.exit(2)
    except Exception as e:
        traceback.print_exc()
        say(f"@@ERROR shot {i}: {e}")
        sys.exit(1)

say(f"@@FINISHED {made} {skipped} {time.time() - t_start:.0f}")
