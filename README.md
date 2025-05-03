# KLTN

## Setup conda
```bash
apt update; apt install curl -y; curl -sSL https://raw.githubusercontent.com/hienhayho/scripts/refs/heads/main/scripts/intall_fzf.sh | bash; curl -sSL https://raw.githubusercontent.com/hienhayho/scripts/refs/heads/main/scripts/setup_miniconda_linux.sh | bash; source ~/.bashrc
```

## Download data

```bash
conda create -n ms python=3.10 -y
conda activate ms

pip install gdown

gdown --fuzzy https://drive.google.com/file/d/1leyj7dF_-CJlvguI-CRUVldlg50teBov/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1IZRRvPpfWNk9qRwigvTyiEzZPf5xl7a-/view?usp=sharing
```

## Install ms-swift

```bash
conda install nvidia/label/cuda-12.2.0::cuda-toolkit

cd ms-swift

pip install -v -e .

pip install wandb deepspeed
```

## Run SFT
**1. Fill `HF_TOKEN` and `WANDB_API_KEY` in `sft_qwen2_5_3b.sh` and `sft_sea_llm_kltn.sh`**

**2. Qwen2.5-7B-Instruct**

- Run on `xml` data:
```bash
bash sft_qwen2_5_3b.sh Qwen/Qwen2.5-7B-Instruct data_xml.jsonl qwen2_5_7b_new_data_xml
```

- Run on `yaml` data:
```bash
bash sft_qwen2_5_3b.sh Qwen/Qwen2.5-7B-Instruct data_yaml.jsonl qwen2_5_7b_new_data_yaml
```

**3. SeaLLMs-v3-7B-Chat**
- Run on `xml` data:
```bash
bash sft_sea_llm_kltn.sh SeaLLMs/SeaLLMs-v3-7B-Chat data_xml.jsonl seallm_7b_new_data_xml 
```

- Run on `yaml` data:
```bash
bash sft_sea_llm_kltn.sh SeaLLMs/SeaLLMs-v3-7B-Chat data_yaml.jsonl seallm_7b_new_data_yaml 
```