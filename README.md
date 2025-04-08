# LAION-C: Benchmarking Model Performance

**LAION-C** is a lightweight Python package for evaluating classifier robustness on the LAION-C dataset using **superclass-level Top-1 accuracy**.

---

## 🚀 Quickstart

### 📥 Dataset Download

🔗 You can download the LAION-C dataset from the Zenodo link [here](https://zenodo.org/records/14051887)！ 

---

### 🔧 Installation

Simply clone the repository and install the package (requires Python 3.8+):

1. You can set the dataset location via an environment variable from the command line:
   ```bash
   export LAIONC_DATASET_DIR=/path/to/dataset
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/FanfeiLi/LAION-C.git
   ```
3. Install the package in editable mode:
   ```bash
   cd LAION-C
   pip install -e .
   ```

### Highest OOD (out-of-distribution) distortion robustness

|Rank| Model Name           | Avg. Accuracy (%) |
|----|----------------------|-------------------|
| 1🏅️ | EVA-G-P14-560-M30M-IN22K        | 67.5   |
| 2🥈 | EVA02-L-P14-448-MIM-M38M-IN22K  | 66.8   |
| 3🥉 | ViT-L-P14-224-CLIP-OpenAI-IN12K | 57.8   |
| 4  | ViT-H-P14-336-CLIP-LAION-IN12K  | 57.3   |
| **5**  | **Best Human Observer**             | **55.2**  |
| 6  | ConvNeXt-XXL-CLIP-LAION-IN1K    | 54.8   |
| 7  | GPT-4o                          | 54.1   |
| 8  | Gemini 1.5 Pro                  | 50.2   |
| 9  | BEiT-v2-L-P16-224-IN1K          | 47.7   |
| 10 | ViT-B-P16-224-AugReg-IN21K      | 47.1   |

### 📚 Citation

If you use **LAION-C** in your research, please cite the paper:

> _[Citation will be available upon arXiv publication.]_

In the meantime, you can find the submission on **OpenReview** [here](https://openreview.net/forum?id=t1IBHkU2bt&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FWorkshop%2FSCSL%2FAuthors%23your-submissions))!
