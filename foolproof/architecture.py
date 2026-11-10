"""Strict, torch-free validation for the visual decoder builder."""
def validate(value, vocab_size):
    if not isinstance(value, dict) or set(value) != {"version", "width", "heads", "context", "blocks"}:
        raise ValueError("Use the supported version 1 builder configuration.")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unsupported builder version.")
    limits = {"width": (32, 768), "heads": (1, 16), "context": (32, 1024)}
    for key, (low, high) in limits.items():
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"{key} must be a whole number from {low} to {high}.")
    width, heads = value["width"], value["heads"]
    if width % heads or (width // heads) % 2:
        raise ValueError("Width must divide evenly by heads, with an even head width for RoPE.")
    blocks = value["blocks"]
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 16:
        raise ValueError("Choose between 1 and 16 blocks.")
    kinds = []
    for block in blocks:
        if not isinstance(block, dict) or set(block) != {"type"} or block["type"] not in ("transformer", "attention", "feed_forward"):
            raise ValueError("Only Transformer, attention and feed-forward blocks are supported.")
        kinds.append(block["type"])
    if not any(k in ("transformer", "attention") for k in kinds):
        raise ValueError("Add at least one attention or Transformer block to mix token context.")
    params = width * vocab_size + width
    for kind in kinds:
        params += {"transformer":12*width*width+2*width,
                   "attention":4*width*width+width,
                   "feed_forward":8*width*width+width}[kind]
    if params > 350_000_000:
        raise ValueError("This architecture exceeds the builder's 350 million parameter safety limit.")
    return {"n_layer":len(kinds), "n_head":heads, "n_embd":width,
            "block_size":value["context"], "block_types":kinds}, params
