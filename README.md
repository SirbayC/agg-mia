# AGG-MIA

Common pipeline built around 3 MIAs:
- [TraWiC](https://github.com/SirbayC/TraWiC)
- [MIA_Adv](https://github.com/SirbayC/MIA_Adv)
- [EZ-MIA](https://github.com/SirbayC/ez-mia)

Key objective = determine whether some specific file was used during the pre-training of a LLM.

Binary classification evaluation, using as ground truth the **seen** and **unseen** sources, as listed below.

Target model is StarCoder2, with the following variants (and sources for **seen** files):
  - bigcode/starcoder2-3b trained on the-stack-v2-train-smol 
  - bigcode/starcoder2-7b trained on the-stack-v2-train-smol 
  - bigcode/starcoder2-15b trained on the-stack-v2-train-full

To download (smol) seen files:
1. set SWH_TOKEN token in the environment from https://archive.softwareheritage.org/oidc/profile/#tokens.
2. run from root dir `python src/datasets/download_seen.py`.

For **unseen** files, we rely on The Heap, deduplicated against The Stack v2 (exact_duplicates_stackv2 and near_duplicates_stackv2 set to false).
