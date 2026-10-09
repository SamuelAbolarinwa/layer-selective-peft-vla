# Layer-Selective PEFT VLA

This repository explores whether Maximum Mean Discrepancy (MMD)-based layer selection provides a better accuracy and efficiency trade-off for LoRA fine-tuning of the **OpenVLA-7B** model compared to naive uniform-layer LoRA. 

## Experimental Setup
- **Baseline**: Uniform-layer LoRA fine-tuning on OpenVLA-7B across all 32 layers of the language model backbone.
- **Proposed Method**: MMD-selective LoRA fine-tuning on OpenVLA-7B. We hook into the layers, compute the Maximum Mean Discrepancy between source (LIBERO-Spatial) and target (LIBERO-Object) activations, and selectively fine-tune only the top $K=16$ layers with the highest distribution shifts.
- **Dataset**: [LIBERO-Object](https://github.com/Lifelong-Robot-Learning/LIBERO) (specifically the `libero_object_no_noops` RLDS subset).
- **Core Codebase**: [OpenVLA](https://github.com/openvla/openvla)

## Project Structure
- `notebooks/`: The clean, finalized, and executable Kaggle Jupyter Notebooks for the entire pipeline.
  - `01_baseline_training.ipynb`: Performs the standard baseline uniform-LoRA fine-tuning on OpenVLA-7B.
  - `02_mmd_selective_training.ipynb`: Analyzes the layer activations, calculates MMD scores, generates the custom regex pattern for `target_modules`, and launches the selective LoRA fine-tuning.
  - `03_evaluation.ipynb`: The headless EGL evaluation notebook that evaluates the trained adapter checkpoints against the simulated LIBERO-Object suite. Contains PyTorch `float16` and `load_in_4bit` fixes to safely evaluate inside a Kaggle T4 environment.
  - `04_smoke_test.ipynb`: A verification notebook to validate dataset fetching, environment setup, and OpenVLA dependencies.
- `scripts/`: Custom Python scripts used during development and data extraction.
- `archive/`: A git-ignored folder containing raw outputs, exploratory notebooks, discarded runs, and previous workflow logs. 
- `openvla/` & `LIBERO/`: Submodules / local clones of the dependency frameworks.

## Usage
The entire workflow is heavily optimized for execution in Kaggle Notebook environments using T4 or P100 GPUs. 

1. Create a new Kaggle notebook with Internet and GPU enabled.
2. Upload the desired notebook from `notebooks/`.
3. Follow the markdown instructions inside the notebook to mount required inputs (e.g., HuggingFace token, model checkpoints) and execute the cells.

## Methodology & Findings
We successfully patched OpenVLA's deep `flash_attention_2` and `bfloat16` assumptions to run on cheaper T4 GPUs (using `sdpa` and `float16`), and fixed PEFT's strict matching logic to permit selective `target_modules` tuning. By using MMD, we effectively slashed the required tunable LLM layers by half ($K=16$) resulting in a significantly lower overall parameter budget. The `03_evaluation.ipynb` notebook handles the final rollout benchmarking to assess physical success rates.
