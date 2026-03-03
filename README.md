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
  - bigcode/starcoder2-15b trained on the-stack-v2-train-full (TODO adapt dataset loading script)

For **unseen** files, we rely on The Heap, deduplicated against The Stack v2 (exact_duplicates_stackv2 and near_duplicates_stackv2 set to false).

To download (smol) files:
1. `module load miniconda3 && conda activate /scratch/cosminvasilesc/AGG-MIA/ENV`
1. set SWH_TOKEN token in the environment from https://archive.softwareheritage.org/oidc/profile/#tokens.
1. run from root dir `python src/datasets/download_seen.py`.
1. run from root dir `python src/datasets/download_unseen.py`.
1. run from root dir `python src/datasets/show_parquet_samples.py`.

## Development

1. make local changes
1. sync and submit with `./submit_delftblue.sh`
1. see latest logs with `tail -f "$(ls -t | head -n 1)"`
1. copy outputs with `scp -r cosminvasilesc@login.delftblue.tudelft.nl:/scratch/cosminvasilesc/AGG-MIA/outputs/runs/* "C:\Coding_projects\AGG_MIA\output_runs" && ssh delftblue "rm -rf /scratch/cosminvasilesc/AGG-MIA/outputs/runs/*"`