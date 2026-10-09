import json

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

cells = [
    create_markdown_cell("# OpenVLA-7B MMD-Selective LoRA Fine-Tuning\n### **Phase 2: MMD-Selective LoRA**"),
    
    create_markdown_cell("### 1. Safely Install Pinned Dependencies (Using Python Lists)"),
    create_code_cell("""import subprocess
import sys

def install(packages, no_deps=False):
    cmd = [sys.executable, "-m", "pip", "install"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.extend(packages)
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# OpenVLA compatibility pins
install([
    "transformers==4.40.1",
    "tokenizers==0.19.1",
    "timm==0.9.10",
    "peft==0.11.1",
    "accelerate>=0.25.0",
    "sentencepiece==0.1.99",
    "draccus==0.8.0",
    "json-numpy",
    "protobuf",
])

# Standard utilities
install([
    "einops",
    "jsonlines",
    "rich",
    "matplotlib",
    "huggingface_hub",
    "wandb",
])

# IMPORTANT:
# Use a newer bitsandbytes compatible with Kaggle's environment.
install([
    "bitsandbytes>=0.45.0",
])

# TensorFlow Datasets
install([
    "tensorflow_datasets==4.9.3",
])

# DLIMP dataloader
install([
    "git+https://github.com/moojink/dlimp_openvla"
], no_deps=True)

print("\\n✅ All installations completed successfully!")"""),

    create_markdown_cell("### 2. Configure Global Paths and Install OpenVLA Package"),
    create_code_cell("""import os
import subprocess
import sys

WORKDIR = "/kaggle/working"
OPENVLA_DIR = os.path.join(WORKDIR, "openvla")
DATA_DIR = os.path.join(WORKDIR, "datasets", "openvla", "modified_libero_rlds")
CHECKPOINT_DIR = os.path.join(WORKDIR, "mmd_selective_lora_run")

if not os.path.exists(OPENVLA_DIR):
    print("Cloning OpenVLA...")
    subprocess.run(["git", "clone", "https://github.com/openvla/openvla", OPENVLA_DIR], check=True)
else:
    print("✅ OpenVLA already cloned.")

# Install OpenVLA natively into the Python environment
subprocess.run([sys.executable, "-m", "pip", "install", "-e", OPENVLA_DIR, "--no-deps"], check=True)
print("✅ Installed OpenVLA package.")"""),

    create_markdown_cell("### 3. Patch Unnecessary DROID Dependencies (`tensorflow_graphics`)"),
    create_code_cell("""patch_file = os.path.join(OPENVLA_DIR, "prismatic", "vla", "datasets", "rlds", "oxe", "utils", "droid_utils.py")
with open(patch_file, "r") as f:
    content = f.read()

if "import tensorflow_graphics.geometry.transformation as tfg" in content:
    content = content.replace("import tensorflow_graphics.geometry.transformation as tfg", 
                              "# import tensorflow_graphics.geometry.transformation as tfg")
    with open(patch_file, "w") as f:
        f.write(content)
    print("✅ Patched droid_utils.py to remove tensorflow_graphics dependency.")
else:
    print("✅ droid_utils.py already patched.")"""),

    create_markdown_cell("### 4. Rigorous Preflight Version Audit & Deep Import Check"),
    create_code_cell("""import sys
from pathlib import Path
import torch
import transformers
import peft
import timm
import tensorflow as tf
import tensorflow_datasets as tfds
import bitsandbytes as bnb
from dlimp import DLataset

# MAKE OPENVLA SOURCE TREE VISIBLE TO PYTHON
OPENVLA_DIR = Path("/kaggle/working/openvla")
if str(OPENVLA_DIR) not in sys.path:
    sys.path.insert(0, str(OPENVLA_DIR))

# VERSION AUDIT
print("\\n--- VERSION AUDIT ---")
print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__} (CUDA: {torch.version.cuda})")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT:         {peft.__version__}")
print(f"BitsAndBytes: {bnb.__version__}")
print(f"TensorFlow:   {tf.__version__}")

if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"GPUs Detected: {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f" - GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1024**3:.2f} GB)")
else:
    raise RuntimeError("❌ No GPU found! Please enable a GPU accelerator in Kaggle.")"""),

    create_markdown_cell("### 5. Download the LIBERO-Object and LIBERO-Spatial RLDS Datasets"),
    create_code_cell("""import os
import subprocess

os.makedirs(DATA_DIR, exist_ok=True)
print("Downloading Target Dataset (LIBERO Object)...")
subprocess.run([
    "huggingface-cli", "download", "openvla/modified_libero_rlds",
    "--repo-type", "dataset",
    "--include", "libero_object_no_noops/*",
    "--local-dir", DATA_DIR
], check=True)

print("Downloading Source Proxy Dataset (LIBERO Spatial)...")
subprocess.run([
    "huggingface-cli", "download", "openvla/modified_libero_rlds",
    "--repo-type", "dataset",
    "--include", "libero_spatial_no_noops/*",
    "--local-dir", DATA_DIR
], check=True)
print("✅ Datasets downloaded.")"""),

    create_markdown_cell("### 6. Patch Accelerate default config AND big_modeling.py"),
    create_code_cell("""import os
import accelerate

# Create default_config.yaml
accelerate_config = \"\"\"compute_environment: LOCAL_MACHINE
debug: false
distributed_type: NO
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
\"\"\"

os.makedirs(os.path.expanduser('~/.cache/huggingface/accelerate'), exist_ok=True)
with open(os.path.expanduser('~/.cache/huggingface/accelerate/default_config.yaml'), 'w') as f:
    f.write(accelerate_config)

# Patch big_modeling.py
accelerate_dir = os.path.dirname(accelerate.__file__)
accelerate_file = os.path.join(accelerate_dir, "big_modeling.py")

with open(accelerate_file, "r") as f:
    source = f.read()

old_block = \"\"\"    if len(set(device_map.values())) == 1:
        if main_device is None:
            main_device = list(device_map.values())[0]
        model.to(main_device)
\"\"\"

new_block = \"\"\"    if len(set(device_map.values())) == 1:
        if main_device is None:
            main_device = list(device_map.values())[0]

        # Do not call .to() on bitsandbytes quantized models.
        if not getattr(model, "is_quantized", False):
            model.to(main_device)
\"\"\"

if old_block in source:
    source = source.replace(old_block, new_block)
    with open(accelerate_file, "w") as f:
        f.write(source)
    print("✅ Patched Accelerate successfully.")
elif new_block in source:
    print("✅ Accelerate is already patched.")
else:
    raise RuntimeError("Could not find the expected Accelerate dispatch_model() code block.")"""),

    create_markdown_cell("### 7. Create `compute_mmd.py` Script"),
    create_code_cell("""%%writefile compute_mmd.py
import os
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoConfig, AutoModelForVision2Seq, BitsAndBytesConfig, AutoProcessor
from prismatic.vla.materialize import get_vla_dataset_and_collator
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

def rbf_kernel(x, y, bandwidth):
    dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-dist / (2 * bandwidth ** 2))

def compute_mmd(x, y):
    combined = torch.cat([x, y], dim=0)
    pdist = torch.pdist(combined, p=2)
    bandwidth = torch.median(pdist) if pdist.numel() > 0 else 1.0
    bandwidth = max(bandwidth, 1e-4)
    
    xx = rbf_kernel(x, x, bandwidth)
    yy = rbf_kernel(y, y, bandwidth)
    xy = rbf_kernel(x, y, bandwidth)
    
    m, n = x.size(0), y.size(0)
    xx_sum = (torch.sum(xx) - torch.trace(xx)) / (m * (m - 1))
    yy_sum = (torch.sum(yy) - torch.trace(yy)) / (n * (n - 1))
    xy_sum = torch.sum(xy) / (m * n)
    
    return max(0.0, xx_sum + yy_sum - 2 * xy_sum).item()

def main():
    num_samples = 64
    device = "cuda"
    
    print("Loading model in 4-bit for MMD scoring...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    
    activations = defaultdict(list)
    def get_hook(layer_idx):
        def hook(module, input, output):
            hidden_states = output[0]
            pooled = hidden_states.mean(dim=1).detach()
            activations[layer_idx].append(pooled)
        return hook
        
    hooks = []
    num_layers = len(model.language_model.model.layers)
    for i in range(num_layers):
        h = model.language_model.model.layers[i].register_forward_hook(get_hook(i))
        hooks.append(h)
        
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    def collect_activations(dataset_name, name):
        print(f"Collecting activations for {name} ({dataset_name})...")
        dataset = RLDSDataset(
            "/kaggle/working/datasets/openvla/modified_libero_rlds",
            dataset_name,
            batch_transform,
            resize_resolution=tuple(model.config.image_sizes),
            shuffle_buffer_size=1000,
            image_aug=False,
        )
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collator)
        
        activations.clear()
        collected = 0
        with torch.no_grad():
            for batch in dataloader:
                if collected >= num_samples:
                    break
                try:
                    model(
                        input_ids=batch["input_ids"].to(device), 
                        attention_mask=batch["attention_mask"].to(device), 
                        pixel_values=batch["pixel_values"].to(torch.float16).to(device), 
                        labels=batch["labels"]
                    )
                    collected += 1
                    if collected % 16 == 0:
                        print(f"  ... {collected}/{num_samples}")
                except Exception as e:
                    print(f"Skipping a batch due to error: {type(e).__name__}: {str(e)}")
                    continue
                    
        for layer_idx in range(num_layers):
            assert len(activations[layer_idx]) > 0, f"CRITICAL FAILURE: Zero activations collected for layer {layer_idx} on dataset {dataset_name}."
            
        return {k: torch.cat(v, dim=0) for k, v in activations.items()}
        
    source_acts = collect_activations("libero_spatial_no_noops", "Source")
    target_acts = collect_activations("libero_object_no_noops", "Target")
    
    for h in hooks:
        h.remove()
        
    print("\\n=== MMD Scores by Layer ===")
    scores = []
    for i in range(num_layers):
        mmd = compute_mmd(source_acts[i].float(), target_acts[i].float())
        scores.append((i, mmd))
        print(f"Layer {i:2d}: MMD = {mmd:.6f}")
        
    scores.sort(key=lambda x: x[1], reverse=True)
    K = 16
    selected = sorted([x[0] for x in scores[:K]])
    
    print(f"\\nTop {K} layers selected by MMD:")
    print(selected)
    
    with open("selected_layers.txt", "w") as f:
        f.write(",".join(map(str, selected)))

if __name__ == "__main__":
    main()"""),

    create_markdown_cell("### 8. Run MMD scoring step"),
    create_code_cell("""!PYTHONPATH=/kaggle/working/openvla python compute_mmd.py"""),

    create_markdown_cell("### 9. Generate `finetune_mmd_selective.py` (with all fixes)"),
    create_code_cell("""import os

# Read selected layers
if os.path.exists("selected_layers.txt"):
    with open("selected_layers.txt", "r") as f:
        selected_layers = f.read().strip()
else:
    raise RuntimeError("selected_layers.txt not found. MMD scoring failed.")
    
layer_pattern = "|".join(selected_layers.split(","))
print(f"Selected layers for LoRA: {selected_layers}")

with open("openvla/vla-scripts/finetune.py", "r") as f:
    content = f.read()

# 1. FP16 patch
content = content.replace("bnb_4bit_compute_dtype=torch.bfloat16", "bnb_4bit_compute_dtype=torch.float16")
content = content.replace("torch_dtype=torch.bfloat16", "torch_dtype=torch.float16")
content = content.replace("dtype=torch.bfloat16", "dtype=torch.float16")
content = content.replace(".to(torch.bfloat16)", ".to(torch.float16)")

# 2. DDP CPU-to-GPU param move before wrapping
target_ddp = "vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)"
device_move_code = '''
    # Manually move any CPU-resident LoRA weights/buffers to the GPU before DDP.
    for param in vla.parameters():
        if param.device.type == 'cpu':
            param.data = param.data.to(device_id)
    for buffer in vla.buffers():
        if buffer.device.type == 'cpu':
            buffer.data = buffer.data.to(device_id)

    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    vla._set_static_graph()  # avoids a related "double-ready" DDP crash
'''
content = content.replace(target_ddp, device_move_code)

# 3. Target modules change
regex_str = r'target_modules=r"^(language_model\\.model\\.layers\\.(' + layer_pattern + r')\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))|(vision_backbone|projector)\\..*(fc1|fc2|fc3|proj|q|kv|qkv)|language_model\\.lm_head)$"'
content = content.replace('target_modules="all-linear"', regex_str)

# 4. Shuffle buffer size reduction
content = content.replace("shuffle_buffer_size: int = 100_000", "shuffle_buffer_size: int = 1_000")

# 5. WANDB disabled
content = content.replace("import wandb", "import wandb\\nimport os\\nos.environ['WANDB_MODE'] = 'disabled'")

with open("openvla/vla-scripts/finetune_mmd_selective.py", "w") as f:
    f.write(content)

print("Created finetune_mmd_selective.py with all fixes applied.")"""),

    create_markdown_cell("### 9.5. Verify Target Modules and Parameter Budget"),
    create_code_cell("""import os
import torch
import re
from transformers import AutoConfig, AutoModelForVision2Seq
from peft import get_peft_model, LoraConfig

print("Verifying K=16 PEFT wrapping budget...")

# Get the regex from selected_layers.txt
with open("selected_layers.txt", "r") as f:
    selected_layers = f.read().strip()
layer_pattern = "|".join(selected_layers.split(","))
regex_str = r'^(language_model\\.model\\.layers\\.(' + layer_pattern + r')\\.(self_attn\\.(q_proj|k_proj|v_proj|o_proj)|mlp\\.(gate_proj|up_proj|down_proj))|(vision_backbone|projector)\\..*(fc1|fc2|fc3|proj|q|kv|qkv)|language_model\\.lm_head)$'

with torch.device("meta"):
    config = AutoConfig.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=regex_str,
    lora_dropout=0.0,
    task_type="CAUSAL_LM",
    init_lora_weights="gaussian",
)

peft_model = get_peft_model(model, lora_config)

trainable_params, all_param = peft_model.get_nb_trainable_parameters()
print(f"\\nTotal Trainable Parameters: {trainable_params}")

# The baseline parameter budget for rank 32 on all-linear (with the verified module list)
# is exactly 110,828,288.
# 16 LLM layers * 2,301,952 parameters = 36,831,232.
# Non-LLM parameters = 110,828,288 - (32 * 2,301,952) = 37,165,824.
# Total expected for K=16: 36,831,232 + 37,165,824 = 73,997,056.
expected = 73997056
if trainable_params == expected:
    print(f"✅ Parameter budget EXACTLY matches the K=16 expectation ({expected}).")
else:
    print(f"❌ MISMATCH: Expected {expected}, got {trainable_params}")

print("\\nWrapped Modules:")
for name, module in peft_model.named_modules():
    if "lora_A" in name:
        print(name.replace(".lora_A.default", ""))
"""),

    create_markdown_cell("### 10. Launch training! (100 steps for verification)"),
    create_code_cell("""import sys
import os
import subprocess

CHECKPOINT_DIR = "/kaggle/working/mmd_selective_lora_run"
adapter_tmp_dir = os.path.join(CHECKPOINT_DIR, "tmp")
DATA_DIR = "/kaggle/working/datasets/openvla/modified_libero_rlds"

cmd = [
    sys.executable,
    "-m",
    "torch.distributed.run",

    "--standalone",
    "--nnodes=1",
    "--nproc-per-node=1",

    "openvla/vla-scripts/finetune_mmd_selective.py",

    "--vla_path",
    "openvla/openvla-7b",

    "--data_root_dir",
    DATA_DIR,

    "--dataset_name",
    "libero_object_no_noops",

    "--run_root_dir",
    CHECKPOINT_DIR,

    "--adapter_tmp_dir",
    adapter_tmp_dir,

    "--use_lora",
    "True",

    "--use_quantization",
    "True",

    "--lora_rank",
    "32",

    "--batch_size",
    "1",

    "--grad_accumulation_steps",
    "1",

    "--learning_rate",
    "5e-4",

    "--image_aug",
    "False",

    "--wandb_project",
    "",

    "--wandb_entity",
    "",

    "--save_steps",
    "100",

    "--max_steps",
    "100",
]

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0"
env["WANDB_MODE"] = "disabled"

print("Starting OpenVLA MMD-Selective LoRA Fine-Tuning...")
print("Command:", " ".join(cmd))

log_path = os.path.join(CHECKPOINT_DIR, "training.log")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

with open(log_path, "w") as log_file:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    for line in process.stdout:
        print(line, end="")
        log_file.write(line)
    process.wait()

if process.returncode != 0:
    print(f"❌ Training failed with exit code {process.returncode}")
    raise RuntimeError(f"Training failed with exit code {process.returncode}")
else:
    print("✅ Training process exited successfully!")"""),

    create_markdown_cell("### 11. Verify Adapter Output"),
    create_code_cell("""import glob
import json

adapters = glob.glob("/kaggle/working/mmd_selective_lora_run/**/adapter_model.safetensors", recursive=True)
configs = glob.glob("/kaggle/working/mmd_selective_lora_run/**/adapter_config.json", recursive=True)

assert adapters, "❌ No LoRA adapter was saved."
assert configs, "❌ No adapter_config.json was saved."

adapter_path = adapters[0]
config_path = configs[0]
print(f"\\n✅ SUCCESS! Found LoRA adapter at: {adapter_path}")

size_mb = os.path.getsize(adapter_path) / (1024 * 1024)
print(f"Adapter Size: {size_mb:.2f} MB")

with open(config_path, "r") as f:
    cfg = json.load(f)
    print(f"LoRA Rank (r): {cfg.get('r')}")
    print(f"Target Modules: {cfg.get('target_modules')}")
""")
]

nb_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("kaggle_openvla_mmd_selective.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=1)

print("Created kaggle_openvla_mmd_selective.ipynb successfully.")
