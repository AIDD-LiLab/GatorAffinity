# <img src="./assets/icon.png" alt="GatorAffinity" width="50" align="center"/> GatorAffinity: Boosting the Protein-Ligand Binding Affinity Prediction with Synthetic Structural Data



A geometric deep learning model for protein-ligand binding affinity prediction, leveraging **unprecedented large-scale synthetic structural data**.

![](./assets/flowchart.png)

## Breakthrough: Synthetic Dataset at Scale

![](./assets/dataset.png)

- **450K+ Kd/Ki complexes** generated using Boltz-1 [[4]](#references) structure prediction 
- **1M+ IC50 complexes** from SAIR database [[1]](#references)  
- **Total: 1.5M synthetic protein-ligand pairs for pre-training**

## Installation

### Environment:
```bash
git clone https://github.com/AIDD-LiLab/GatorAffinity.git
cd GatorAffinity
bash environment.sh
```

### Data Download

#### Original Structural Data
1. **GatorAffinity-DB Complete Original Data**  
    https://huggingface.co/datasets/AIDDLiLab/GatorAffinity-DB

2. **SAIR Complete Original Data**  
    https://www.sandboxaq.com/sair

#### Preprocessed Data
1. **Kd+Ki+IC50 For Pre-training **  
    https://huggingface.co/datasets/AIDDLiLab/Gatoraffinity-Processed-Data

2. **filtered LP-PDBbind For Fine-tuning **  
    https://huggingface.co/datasets/AIDDLiLab/Gatoraffinity-Processed-Data




### Model Checkpoints Download

#### ATOMICA的原子尺度分子相互作用的通用表示模型
我们使用该预训练的表示模型作为GatorAffinity的backbone，模型参数可以在这里下载: https://huggingface.co/ada-f/ATOMICA/tree/main/ATOMICA_checkpoints/pretrain
我们论文的实验表明，在预训练结构数据较少时候，使用该预训练的表示模型可以有效提升模型性能，但收益随着预训练结构的增加而递减

#### GatorAffinity在IC50+Kd+Ki上预训练后的模型

#### GatorAffinity在IC50+Kd+Ki上预训练后并根据LP-PDBbind划分在experimental structure上微调的模型(最优性能)


## Usage




### Training
```bash
python train.py \
    --train_set_path train.pkl \
    --valid_set_path valid.pkl \
    --pretrain_ckpt check_points/epoch6_step1148.ckpt
```

### Inference
```bash
python inference.py \
    --model_ckpt check_points/epoch6_step1148.ckpt \
    --test_set_path test_data/test.pkl
```

### Custom Data Processing
<!-- This section is intentionally left blank for users to customize based on their specific data processing needs -->

## Performance

**State-of-the-art on filtered LP-PDBbind [[2]](#references):**

![](./assets/lp_pdbbind.png)

## Citation

```bibtex
@article{gatoraffinity2025,
  title={GatorAffinity: Boosting the Protein-Ligand Binding Affinity Prediction with Synthetic Structural Data},
  author={Anonymous},
  journal={Arxiv},
  year={2024}
}
```

## References

[1] Lemos, P., Beckwith, Z., Bandi, S., Van Damme, M., Crivelli-Decker, J., Shields, B.J., Merth, T., Jha, P.K., De Mitri, N., Callahan, T.J., et al. (2025). SAIR: Enabling deep learning for protein-ligand interactions with a synthetic structural dataset. *bioRxiv*.

[2] Wang, Y., Sun, K., Li, J., Guan, X., Zhang, O., Bagni, D., Zhang, Y., Carlson, H.A., Head-Gordon, T. (2025). A workflow to create a high-quality protein–ligand binding dataset for training, validation, and prediction tasks. *Digital Discovery*, 4(5), 1209-1220.

[3] Fang, A., Zhang, Z., Zhou, A., and Zitnik, M. (2025). ATOMICA: Learning Universal Representations of Intermolecular Interactions. *bioRxiv*.

[4] Wohlwend, J., Corso, G., Passaro, S., Reveiz, M., Leidal, K., Swiderski, W., Portnoi, T., Chinn, I., Silterra, J., Jaakkola, T., et al. (2024). Boltz-1: Democratizing biomolecular interaction modeling. *bioRxiv*.

## Acknowledgments

This work builds upon the ATOMICA framework [[3]](#references) for learning universal representations of intermolecular interactions. We thank the ATOMICA authors for making their codebase available at https://github.com/mims-harvard/ATOMICA.