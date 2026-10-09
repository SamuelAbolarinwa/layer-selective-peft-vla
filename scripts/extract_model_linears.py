"""Extract all linear module base names from the full OpenVLA HF model.
Uses trust_remote_code=True to let HF load the model class directly 
from the hub, bypassing the local prismatic import chain."""
from transformers import AutoConfig, AutoModelForVision2Seq
from accelerate import init_empty_weights
import torch.nn as nn

config = AutoConfig.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

with init_empty_weights():
    model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)

print("=== NON-LLM LINEAR MODULES (vision backbone + projector + lm_head) ===")
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        if "language_model.model.layers" not in name:
            print(f"  {name}: in={module.in_features}, out={module.out_features}")

print("\n=== BASE NAMES ANALYSIS ===")
llm_base = set()
non_llm_base = set()
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        bn = name.rsplit(".", 1)[-1]
        if "language_model.model.layers" in name:
            llm_base.add(bn)
        else:
            non_llm_base.add(bn)

print(f"LLM-only base names: {sorted(llm_base - non_llm_base)}")
print(f"Non-LLM-only base names: {sorted(non_llm_base - llm_base)}")
print(f"Shared base names: {sorted(llm_base & non_llm_base)}")

# Count total linear modules
total = 0
llm_count = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        total += 1
        if "language_model.model.layers" in name:
            llm_count += 1
print(f"\nTotal linear modules: {total}")
print(f"LLM layer linears: {llm_count}")
print(f"Non-LLM linears: {total - llm_count}")

# Get output embedding name
output_emb = model.get_output_embeddings()
if output_emb is not None:
    for name, module in model.named_modules():
        if module is output_emb:
            print(f"\nOutput embedding (excluded by all-linear): {name}")
            break
