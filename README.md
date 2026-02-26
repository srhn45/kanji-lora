# Kanji LoRA

Fine-tunes Stable Diffusion v1.5 with LoRA to generate kanji images from English meaning prompts (e.g. `"music"`, `"water"`).

Some sample outputs can be found in `generated_kanji/`.

---

## Setup

```bash
pip install torch diffusers peft transformers torchvision accelerate cairosvg lxml Pillow numpy tqdm
```

You'll also need `kanjidic2.xml` (from [EDRDG](https://www.edrdg.org/wiki/index.php/KANJIDIC_Project)) and `kanjivg-all.zip` (from [KanjiVG](https://kanjivg.tagaini.net/)).

---

## Building the Dataset

**1. Extract KanjiVG SVGs**
```bash
python extractzip.py
```
Extracts `kanjivg-all.zip` into `kanji/`.

**2. Convert SVGs to PNGs**
```bash
python svgtopng.py
```
Strips stroke-number annotations from SVGs, then renders clean 256×256 binary PNGs into `kanjivg_png/`.

**3. Build metadata**
```bash
python datasetbuilder.py
```
Pairs each PNG with its English meanings from `kanjidic2.xml` and writes `metadata.jsonl`.

---

## Training

```bash
python train.py \
  --metadata metadata.jsonl \
  --image_folder kanjivg_png \
  --output_dir ./lora-kanji \
  --epochs 100 \
  --batch_size 2 \
  --lr 1e-4
```

Checkpoints are saved every 3 epochs.