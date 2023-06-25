## Machine Learning for Earth Observation
A repository with some examples of machine learning applications in the domain of Earth Observation/remote sensing.
Specifically, the repository is meant for image classification on the EuroSAT dataset [1,2]. The goal is to leverage different packages, in order to check:

1. How easy they are to use
2. The quality of the results, vs. [3]

For each package, a Vision Transformer (ViT) will be fine-tuned on the EuroSAT datasel, as research shows this is the best approach for multi-class image classification [4].

### Installation
```bash
conda create -n earthObs
conda activate earthObs
conda install -c fastchan fastai
pip install requirements.txt
```

(Note that the installation is specific for using miniconda, cfr. https://anaconda.org/fastai/fastai)

### Contents
Fine-tuning for the EuroSAT dataset with:
- Fast.ai: train_fast_vit.ipynb
- Hugging Face (+torch): train_hf_vit.ipynb
- Timm (PyTorch Image Models): train_timm_vit.ipynb
- TensorFlow: TODO

TODO for Fast & Timm: use base ViT model pretrained on ImageNet21k instead of ViT-tiny.

TODO for all: align performance metric with [3], by using 1) stratified sampling and 2) a 60-20-20 separation of the data.

### Conclusion (preliminary)
The best results are through using Hugging Face's transformers package. This package is also relatively easy to use, well-documented and has a wide database of datasets and models.

It should be remarked that higher performance should be achived by using a ViT pre-trained with SSL methods (e.g. DINO, MAE). This was done by Wang et al. [4], but on a multi-spectral, multi-temporal dataset. Again, this should be relatively easy through [Hugging Face](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining#mae).

For multi-specral and multi-temporal data, one should revert to base (Py)Torch/TensorFlow, as standard image processing NN/libraries focus on RGB data. Still, there are some recent interesting works that can be leveraged. One is from Li et al. [4], showing the use of multi-temporal data in combination with transformers (though not multi-spectral), the other from Cong et al. [6]. The latter work is specifically for satellite imagery, acts on multi-temperal and multi-spectral data and provides code which can be reused [7]. The work can possibly enhanced by leveraging the dataset from [4].

### References
[1] P. Helber, B. Bischke, A. Dengel and D. Borth, "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 7, pp. 2217-2226, July 2019, doi: 10.1109/JSTARS.2019.2918242.

[2] P. Helber, B. Bischke, A. Dengel and D. Borth, "Introducing Eurosat: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification," IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, Valencia, Spain, 2018, pp. 204-207, doi: 10.1109/IGARSS.2018.8519248.

[3] Ivica Dimitrovski, Ivan Kitanovski, Dragi Kocev, Nikola Simidjievski, Current trends in deep learning for Earth Observation: An open-source benchmark arena for image classification, ISPRS Journal of Photogrammetry and Remote Sensing, Volume 197, 2023, Pages 18-35, ISSN 0924-2716, https://doi.org/10.1016/j.isprsjprs.2023.01.014. (https://www.sciencedirect.com/science/article/pii/S0924271623000205)

<div class="csl-entry">[4] Wang, Y., Braham, N. A. A., Xiong, Z., Liu, C., Albrecht, C. M., &#38; Zhu, X. X. (2022). <i>SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation</i>. http://arxiv.org/abs/2211.07044</div>

<div class="csl-entry">[5] Li, Z., Li, S., &#38; Yan, X. (2023). <i>Time Series as Images: Vision Transformer for Irregularly Sampled Time Series</i>. https://doi.org/https://doi.org/10.48550/arXiv.2303.12799</div>

<div class="csl-entry">[6] Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D. B., &#38; Ermon, S. (2022). <i>SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery</i>. http://arxiv.org/abs/2207.08051</div>

<div class="csl-entry">[7] <i>GitHub - sustainlab-group/SatMAE: Official code repository for NeurIPS 2022 paper “SatMAE: Pretraining Transformers for Temporal and Multi-Spectral Satellite Imagery.”</i> (n.d.). Retrieved June 24, 2023, from https://github.com/sustainlab-group/SatMAE</div>

Note: references 1-3 should be reformatted with Mendeley.