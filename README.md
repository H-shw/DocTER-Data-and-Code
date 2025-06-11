# DocTER-Data-and-Code

This repository is for the paper "DocTER: Evaluating Document-based Knowledge Editing".

# Requirements
```
datasets==1.18.3
einops==0.4.0
gpustat==1.1
hydra-core==1.1.1
higher==0.2.1
importlib-metadata==6.3.0
matplotlib==3.5.1
nltk==3.6.5
numpy==1.22.1
omegaconf==2.1.1
pandas==1.4.0
PyYAML==6.0
scikit-learn==1.0.2
scipy==1.7.3
sentence-transformers==3.2.1
tokenizers==0.20.3
torch==2.0.1
tqdm==4.62.3
transformers==4.46.2
openai==1.57.0
peft==0.7.1
timm==0.9.7
iopath==0.1.10
opencv-python==4.8.0.76
fairscale==0.4.13
av==14.2.0
qwen_vl_utils==0.0.10
zhipuai==2.1.5.20250415
sentencepiece==0.2.0
```
We also use [EasyEdit](https://github.com/zjunlp/EasyEdit/) for MEMIT implementation. Please install it as well.


# DocTER Dataset

Dataset are available at [here] (https://github.com/H-shw/DocTER-Data-and-Code/tree/main/data). 

```bash
#  structure 
Data structure.
```bash
#  structure 
├─ docs
   ├─ edit_sucess_doc.json # Documents data for evaluating Edit Sucess, Locality and Reasoning.
   ├─ cross_lingual_en_doc.json # Documents data for evaluating Cross-lingual Editing (en->zh).
   ├─ cross_lingual_zh_doc.json # Documents data for evaluating Cross-lingual Editing (zh->en).
├─ test_data   # Test data for evaluation
   ├─ edit_success_eval.json  # Test data for evaluating Edit Sucess perspective.
   ├─ locality_eval.json  # Test data for evaluating Locality perspective.
   ├─ reasoning_eval.json  # Test data for evaluating Reasoning perspective.
   ├─ cross_zh_eval.json  # Test data for evaluating Cross-lingual Editing (en->zh) perspective.
   ├─ cross_en_eval.json  # Test data for evaluating Cross-lingual Editing (zh->en) perspective.
```

# Evaluation
See `scripts/`
```bash
#  structure 
Data structure.
```bash
#  structure 
├─ scripts   # Test data for evaluation
   ├─ eval_base.json  # script to evaluate base model or edited model (input the model path).
   ├─ run_finetune.sh  # Training script for FT.
   ├─ run_ike.sh  # Script for IKE.
   ├─ run_memit.sh  # Script for MEMIT.
   ├─ run_searc.sh  # Script for SEARC.
```
