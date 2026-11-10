"""Brain sizes and the starter library of datasets.

Every dataset in STARTER was checked against the live Hugging Face API: public,
not locked, readable as parquet, with a stated open licence.
"""
from __future__ import annotations



def count_params(n_layer: int, n_embd: int, vocab_size: int) -> int:
    """Parameter count of the model without building it (no PyTorch import needed here)."""
    per_block = 12 * n_embd * n_embd + 2 * n_embd
    return n_layer * per_block + vocab_size * n_embd + n_embd

# ----------------------------------------------------------------------------- brain sizes

PRESETS = [
    {"id": "pocket", "name": "Pocket", "n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 128,
     "batch": 32, "lr": 1e-3,
     "blurb": "Runs on any computer, no graphics card needed. Learns the basics of your text: "
              "spelling, spacing, short common words."},
    {"id": "tiny", "name": "Tiny", "n_layer": 6, "n_head": 6, "n_embd": 192, "block_size": 256,
     "batch": 32, "lr": 1e-3,
     "blurb": "Fine on a laptop if you give it an hour. Real words and short phrases."},
    {"id": "small", "name": "Small", "n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 256,
     "batch": 64, "lr": 1e-3,
     "blurb": "The best first choice if you have a graphics card. Whole sentences in the style of your text."},
    {"id": "medium", "name": "Medium", "n_layer": 8, "n_head": 8, "n_embd": 512, "block_size": 512,
     "batch": 48, "lr": 6e-4,
     "blurb": "Wants a graphics card with 6 GB or more, and tens of millions of pieces of text to learn from."},
    {"id": "large", "name": "Large", "n_layer": 12, "n_head": 12, "n_embd": 768, "block_size": 1024,
     "batch": 32, "lr": 6e-4,
     "blurb": "The size of the original GPT-2. Wants 12 GB or more, a lot of text, and many hours."},
    {"id": "huge", "name": "Huge", "n_layer": 24, "n_head": 16, "n_embd": 1024, "block_size": 1024,
     "batch": 32, "lr": 3e-4,
     "blurb": "A 24 GB graphics card, gigabytes of text and days of training. For the patient."},
]
PRESET_BY_ID = {p["id"]: p for p in PRESETS}
ORDER = [p["id"] for p in PRESETS]


def body_params(p: dict) -> int:
    return count_params(p["n_layer"], p["n_embd"], 0)


def train_memory_gb(p: dict, vocab: int, micro_batch: int, amp: bool = True) -> float:
    """Rough GPU memory needed to train (weights + optimiser + activations + output scores)."""
    params = count_params(p["n_layer"], p["n_embd"], vocab)
    weights = params * 16  # weights, gradients, two optimiser moments (4 bytes each)
    per_token = p["n_layer"] * p["n_embd"] * 34 * (2 if amp else 4) + vocab * 10
    return (weights + per_token * p["block_size"] * micro_batch) / 2**30 + 0.6


def dropout_for(n_tokens: int) -> float:
    if n_tokens < 3_000_000:
        return 0.2
    if n_tokens < 30_000_000:
        return 0.1
    return 0.0


def biggest_worth_training(n_tokens: int) -> str:
    """The largest ready-made size this much text can teach without it simply memorising."""
    if n_tokens < 1_500_000:
        return "small"
    if n_tokens < 25_000_000:
        return "medium"
    if n_tokens < 300_000_000:
        return "large"
    return "huge"


def recommend(hw: dict, n_tokens: int, vocab: int) -> dict:
    """Which brain size to suggest, and a fit label for each one."""
    data_max = biggest_worth_training(n_tokens)

    dev = hw.get("device", "cpu")
    vram = float(hw.get("vram_gb") or 0)
    if dev == "cuda":
        hw_max = ("tiny" if vram < 3 else "small" if vram < 6 else "medium" if vram < 10
                  else "large" if vram < 20 else "huge")
    elif dev == "mps":
        hw_max = "small" if vram < 12 else "medium"
    else:
        hw_max = "pocket" if (hw.get("cores") or 1) < 8 else "tiny"

    rec = ORDER[min(ORDER.index(data_max), ORDER.index(hw_max))]
    fits = {}
    for p in PRESETS:
        i = ORDER.index(p["id"])
        note, level = "", "ok"
        if dev == "cpu":
            if i > ORDER.index("small"):
                level, note = "no", "Far too slow without a graphics card."
            elif i > ORDER.index(hw_max):
                level, note = "slow", "Will work, but slowly on this computer."
        else:
            amp = bool(hw.get("amp")) or dev == "mps"
            need_min = train_memory_gb(p, vocab, 1, amp)
            need_ok = train_memory_gb(p, vocab, 8, amp)
            if need_min > vram * 0.92:
                level, note = "no", f"Needs about {need_min:.0f} GB of graphics memory; you have {vram:.0f} GB."
            elif need_ok > vram * 0.92:
                level, note = "slow", "Will just about fit, so it trains in small slow steps."
        if level != "no" and i > ORDER.index(data_max):
            level = "slow" if level == "ok" else level
            note = (note + " " if note else "") + "Bigger than your text needs: it may just memorise it."
        fits[p["id"]] = {"level": level, "note": note.strip()}
    return {"recommended": rec, "fits": fits}


def fit_custom(hw: dict, n_tokens: int, vocab: int, model: dict, params: int) -> dict:
    """The same fit label the ready-made sizes get, for a design built by hand.

    model: what architecture.validate returns (n_embd, block_size, block_types).
    Also says which ready-made size is nearest, so its learning speed and batch size are used.
    """
    import math

    kinds = model.get("block_types") or ["transformer"] * int(model.get("n_layer") or 1)
    # how many full layers' worth of work each token costs (attention is the lighter half)
    layers = sum({"transformer": 1.0, "attention": 0.4, "feed_forward": 0.6}[k] for k in kinds)
    shape = {"n_layer": max(1, math.ceil(layers)), "n_embd": int(model["n_embd"]), "block_size": int(model["block_size"])}
    sized = {q["id"]: count_params(q["n_layer"], q["n_embd"], vocab) for q in PRESETS}
    nearest = min(ORDER, key=lambda i: abs(math.log(sized[i]) - math.log(max(1, params))))
    rec = recommend(hw, n_tokens, vocab)
    dev = hw.get("device", "cpu")
    vram = float(hw.get("vram_gb") or 0)
    level, note = "ok", ""
    if dev == "cpu":
        if params > 1.5 * sized["small"]:
            level, note = "no", "Far too slow without a graphics card. Use a smaller width or fewer blocks."
        elif params > 1.5 * sized[rec["recommended"]]:
            level, note = "slow", "Will work, but slowly on this computer."
    else:
        amp = bool(hw.get("amp")) or dev == "mps"
        weights = params * 16
        per_token = layers * shape["n_embd"] * 34 * (2 if amp else 4) + vocab * 10
        need = lambda micro: (weights + per_token * shape["block_size"] * micro) / 2**30 + 0.6  # noqa: E731
        if need(1) > vram * 0.92:
            level, note = "no", f"Needs about {need(1):.0f} GB of graphics memory; you have {vram:.0f} GB. Use a smaller width, a shorter memory or fewer blocks."
        elif need(8) > vram * 0.92:
            level, note = "slow", "Will just about fit, so it trains in small slow steps."
    if level != "no" and params > 1.5 * sized[biggest_worth_training(n_tokens)]:
        level = "slow" if level == "ok" else level
        note = (note + " " if note else "") + "Bigger than your text needs: it may just memorise it."
    return {"level": level, "note": note.strip(), "params": int(params), "preset": nearest,
            "preset_name": PRESET_BY_ID[nearest]["name"]}


def preset_cards(vocab: int) -> list[dict]:
    out = []
    for p in PRESETS:
        out.append({**p, "params": count_params(p["n_layer"], p["n_embd"], vocab),
                    "body_params": body_params(p)})
    return out


# ----------------------------------------------------------------------------- time budgets

BUDGETS = [
    {"id": "5m", "seconds": 300, "name": "5 minutes", "blurb": "A quick try"},
    {"id": "30m", "seconds": 1800, "name": "30 minutes", "blurb": "A coffee break"},
    {"id": "2h", "seconds": 7200, "name": "2 hours", "blurb": "An afternoon"},
    {"id": "8h", "seconds": 28800, "name": "8 hours", "blurb": "Overnight"},
    {"id": "inf", "seconds": 0, "name": "Until I stop it", "blurb": "No time limit"},
]


# ----------------------------------------------------------------------------- starter library

STARTER = [
    {"key": "shakespeare", "group": "start", "title": "Shakespeare", "chat": False,
     "blurb": "All of Shakespeare's plays in one small file. Already on your computer, so it works "
              "without internet. Trains in minutes: the classic first experiment.",
     "size": "1 MB", "approx_tokens": 1_100_000, "licence": "Public domain",
     "source": {"type": "builtin", "id": "shakespeare"}},
    {"key": "tinystories", "group": "start", "title": "TinyStories", "chat": False,
     "blurb": "Two million very simple children's stories. Even a small AI learns to write "
              "proper sentences from it, which makes it the best first download.",
     "size": "2 GB", "approx_tokens": 470_000_000, "licence": "CDLA-Sharing-1.0",
     "source": {"type": "hf", "id": "roneneldan/TinyStories", "config": "default", "split": "train"}},

    {"key": "wikitext2", "group": "write", "title": "Wikipedia sampler", "chat": False,
     "blurb": "A small slice of good Wikipedia articles. Quick to download; factual, grown-up writing.",
     "size": "11 MB", "approx_tokens": 2_800_000, "licence": "CC BY-SA 3.0",
     "source": {"type": "hf", "id": "Salesforce/wikitext", "config": "wikitext-2-raw-v1", "split": "train"}},
    {"key": "simplewiki", "group": "write", "title": "Simple English Wikipedia", "chat": False,
     "blurb": "The whole encyclopedia written in easy English. 240,000 short articles.",
     "size": "1 GB", "approx_tokens": 230_000_000, "licence": "CC BY-SA 3.0",
     "source": {"type": "hf", "id": "wikimedia/wikipedia", "config": "20231101.simple", "split": "train"}},
    {"key": "wikilang", "group": "write", "title": "Wikipedia in your language", "chat": False,
     "blurb": "Wikipedia in any of 40 languages, so your AI learns to write in yours.",
     "size": "varies", "approx_tokens": 0, "licence": "CC BY-SA 3.0", "languages": True,
     "source": {"type": "hf", "id": "wikimedia/wikipedia", "config": "20231101.{lang}", "split": "train"}},
    {"key": "books", "group": "write", "title": "Classic books", "chat": False,
     "blurb": "48,000 out-of-copyright books from Project Gutenberg: novels, plays, essays.",
     "size": "22 GB", "approx_tokens": 5_500_000_000, "licence": "MIT (texts: public domain)",
     "source": {"type": "hf", "id": "sedthh/gutenberg_english", "config": "default", "split": "train"}},
    {"key": "finewebedu", "group": "write", "title": "Educational web pages", "chat": False,
     "blurb": "Millions of the web's most informative pages. What serious small language models "
              "are trained on. Enormous: you will use a slice of it.",
     "size": "48 GB", "approx_tokens": 10_000_000_000, "licence": "ODC-By",
     "source": {"type": "hf", "id": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "split": "train"}},

    {"key": "everyday", "group": "chat", "title": "Everyday small talk", "chat": True,
     "blurb": "2,000 short, simple conversations. Tiny, so it trains fast: your AI learns "
              "greetings and simple replies.",
     "size": "5 MB", "approx_tokens": 600_000, "licence": "Apache-2.0",
     "source": {"type": "hf", "id": "HuggingFaceTB/everyday-conversations-llama3.1-2k",
                "config": "default", "split": "train_sft"}},
    {"key": "alpaca", "group": "chat", "title": "Questions and answers", "chat": True,
     "blurb": "52,000 questions with answers on all sorts of topics. Your AI learns the shape "
              "of answering a question.",
     "size": "40 MB", "approx_tokens": 10_000_000, "licence": "CC BY 4.0",
     "source": {"type": "hf", "id": "yahma/alpaca-cleaned", "config": "default", "split": "train"}},
    {"key": "smoltalk", "group": "chat", "title": "Big conversation mix", "chat": True,
     "blurb": "460,000 conversations made for teaching small AIs to chat. Needs a graphics card and patience.",
     "size": "1.9 GB", "approx_tokens": 450_000_000, "licence": "Apache-2.0",
     "source": {"type": "hf", "id": "HuggingFaceTB/smol-smoltalk", "config": "default", "split": "train"}},

    {"key": "pycode", "group": "code", "title": "Python coding helper", "chat": True,
     "blurb": "Half a million coding requests with Python answers. Your AI learns to reply with code.",
     "size": "1.5 GB", "approx_tokens": 400_000_000, "licence": "MIT",
     "source": {"type": "hf", "id": "jtatman/python-code-dataset-500k", "config": "default", "split": "train"}},
]

GROUPS = [
    {"id": "start", "title": "Start here"},
    {"id": "write", "title": "Teach it to write"},
    {"id": "chat", "title": "Teach it to chat"},
    {"id": "code", "title": "Teach it to code"},
]

WIKI_LANGS = [
    ("ar", "Arabic"), ("bn", "Bengali"), ("bg", "Bulgarian"), ("ca", "Catalan"), ("zh", "Chinese"),
    ("hr", "Croatian"), ("cs", "Czech"), ("da", "Danish"), ("nl", "Dutch"), ("en", "English (full)"),
    ("et", "Estonian"), ("fi", "Finnish"), ("fr", "French"), ("de", "German"), ("el", "Greek"),
    ("he", "Hebrew"), ("hi", "Hindi"), ("hu", "Hungarian"), ("id", "Indonesian"), ("it", "Italian"),
    ("ja", "Japanese"), ("ko", "Korean"), ("lt", "Lithuanian"), ("ms", "Malay"), ("no", "Norwegian"),
    ("fa", "Persian"), ("pl", "Polish"), ("pt", "Portuguese"), ("ro", "Romanian"), ("ru", "Russian"),
    ("sr", "Serbian"), ("sk", "Slovak"), ("sl", "Slovenian"), ("es", "Spanish"), ("sw", "Swahili"),
    ("sv", "Swedish"), ("ta", "Tamil"), ("th", "Thai"), ("tr", "Turkish"), ("uk", "Ukrainian"),
    ("ur", "Urdu"), ("vi", "Vietnamese"),
]


def default_token_budget(hw: dict) -> int:
    """How many tokens of a huge dataset to fetch by default: enough to be useful on this machine."""
    dev = hw.get("device", "cpu")
    vram = float(hw.get("vram_gb") or 0)
    if dev == "cpu":
        return 10_000_000
    if dev == "mps" or vram < 6:
        return 50_000_000
    if vram < 12:
        return 150_000_000
    return 300_000_000
