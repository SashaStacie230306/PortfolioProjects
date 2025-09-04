# Gen AI Challenge — Storyboard Generation (ComfyUI + WebUI + LLM Prompting)

*Folder: `gen_ai_challenge`*

## What this shows (my 60-sec pitch)

I built a tiny, reproducible pipeline to **generate storyboards** and keep **character consistency** across frames using:

* **ComfyUI** (node graph with a saved workflow),
* **Automatic1111 / WebUI**, and
* an **LLM (Ollama)** to iteratively **improve prompts**.

I included first tests and tuned runs (before/after) plus a lightweight template for keeping the same character through multiple shots.

---

## Folder contents

```
gen_ai_challenge/
├─ 1-storyboard-generation-comfyui.ipynb          # my ComfyUI workflow notes + batch generation
├─ 2-storyboard-generation-webui.ipynb            # the same idea using A1111/WebUI
├─ 3-improve-prompt-ollama.ipynb                  # LLM loop to tighten prompts & negatives
├─ 4-student-template-character-consistency.ipynb # seed + identity anchors + shot list
├─ workflow_juggernaut_xl.json                    # ComfyUI graph (import this)
├─ my_first_test_comfyui.png
├─ my_first_test_webui.png
├─ test_comfyui.png
└─ test_webui.png
```

---

## How my ComfyUI workflow is set up

* **Base model**: `juggernautXL_v9Rdphoto2Lightning.safetensors` (CheckpointLoaderSimple).
* **Sampler**: `KSampler` with `sampler_name="dpmpp_sde"` and `scheduler="karras"`.
* **Speed settings**: `steps=5`, `cfg=2` (Lightning-friendly for fast iterations).
* **Canvas**: `1024×1024` latent, `batch_size=1`.
* **Text**: separate **positive** and **negative** CLIP encoders; **VAE Decode** → **SaveImage**.&#x20;

> Import `workflow_juggernaut_xl.json` in ComfyUI → drop your prompt/negative → press Queue. (I keep a fixed **seed** for character consistency across frames.)&#x20;

---

## What I actually did

* **ComfyUI path**: started from the Lightning Juggernaut XL workflow above, generated first frames (`my_first_test_comfyui.png`), then tightened prompts + negatives and re-ran (`test_comfyui.png`).&#x20;
* **WebUI path**: mirrored the same scenes in Automatic1111 (`my_first_test_webui.png` → `test_webui.png`) to verify results weren’t tool-specific.
* **LLM prompting**: used **Ollama** to iterate on the prompt (“shot list” → “per-frame rich prompt” → “negatives”) in `3-improve-prompt-ollama.ipynb`.
* **Character consistency**: created a small template notebook (`4-student-template-character-consistency.ipynb`) that locks **seed**, reuses a **character anchor** phrase, and drives a **shot list** (wide, mid, close-up, action).

---

## Reproduce 

### ComfyUI (recommended for the workflow file)

1. Install ComfyUI and place the **Juggernaut XL Lightning** checkpoint in your models folder.
2. Launch ComfyUI → **Load** `workflow_juggernaut_xl.json`.
3. Edit:

   * Positive prompt: `heroic teenage girl with a red jacket, curly hair, freckles…`
   * Negative prompt: `blurry, extra fingers, deformed, lowres…`
   * (Optional) **seed** and `steps/cfg` if you want more detail vs. speed.
4. **Queue Prompt** → image saved via `SaveImage`.&#x20;

### WebUI

Open `2-storyboard-generation-webui.ipynb` for the exact text prompts I used. Match sampler/scheduler to the ComfyUI run where possible.

### Prompt improvement with Ollama

Run `3-improve-prompt-ollama.ipynb`:

* give the scene brief + style target,
* let the LLM produce: **(a)** enriched prompt, **(b)** negative list, **(c)** camera terms (wide/mid/close), and
* paste back into ComfyUI/WebUI for the “after” images.

---

## Results (what a reviewer should look at)

* **Before vs. After**: `my_first_test_*` → `test_*` images to see how prompt refinement and negatives tightened composition and artifacts.
* **Consistency**: the character stays recognizable across shots when **seed + anchor prompt** are reused.

---

## Notes on parameters

* Lightning models prefer **low steps** and **low CFG** for speed; the graph ships with `steps=5, cfg=2` to iterate fast, then I bump these slightly if I need more detail.&#x20;
* The sampler/scheduler pair `dpmpp_sde + karras` gave me stable frames for this style.&#x20;

---

## What I’d improve next

* Add a **character embedding/LoRA** pass for even stronger identity lock.
* Script **batch shot lists** (wide → mid → close), saving metadata for each frame.
* Integrate **face/detail control** nodes to reduce minor drift shot-to-shot.

---

## Attribution & safety

* Checkpoint and UI tools are third-party; follow their licenses.
* Keep prompts & outputs appropriate; avoid generating sensitive or harmful content.
