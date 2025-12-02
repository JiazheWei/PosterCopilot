# PosterCopilot

<div align="center">

## Toward Layout Reasoning and Controllable Editing for Professional Graphic Design

[Jiazhe Wei](https://jiazhewei.github.io/)<sup>1,*</sup>, 
[Ken Li](https://kiyotakali.github.io/)<sup>1,*</sup>, 
Tianyu Lao<sup>2</sup>, 
[Haofan Wang](https://haofanwang.github.io/)<sup>2</sup>, 
[Liang Wang](https://scholar.google.com/citations?user=8kzzUboAAAAJ&hl=en)<sup>1,3</sup>, 
[Caifeng Shan](https://caifeng-shan.github.io/)<sup>1</sup>, 
[Chenyang Si](https://chenyangsi.top/)<sup>1,†</sup>

<sup>1</sup>[PRLab, Nanjing University](https://prlab-nju.com/), 
<sup>2</sup>[LibLib.ai](https://www.lovart.ai/zh), 
<sup>3</sup>[Institute of Automation, Chinese Academy of Sciences](http://www.cripac.ia.ac.cn/CN/model/index.htm)

<sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Author

---

[📄 Paper](#) | [🌐 Project Page](#) | [🎬 Demo Video](#)

</div>

---

## 🌟 Highlights

**PosterCopilot** is a cutting-edge framework that advances layout reasoning and controllable editing for professional graphic design using Large Multimodal Models (LMMs).

### Core Features

- **🎯 Geometrically Accurate Layouts**  
  Achieves precise spatial positioning through a progressive three-stage training strategy that moves beyond simple regression to distribution-based learning

- **🎨 Aesthetic Reasoning**  
  Instills human-like design principles and aesthetics through reinforcement learning from aesthetic feedback

- **✂️ Layer-level Control**  
  Enables precise, fine-grained editing of individual layers while maintaining global visual consistency

- **🔄 Multi-round Iterative Editing**  
  Supports professional iterative design workflows with multiple refinement rounds on specific elements

- **🎭 Versatile Applications**  
  Handles complete layout generation, insufficient assets synthesis, theme switching, and canvas reframing

### Three-Stage Training Paradigm

1. **Perturbed Supervised Fine-Tuning (PSFT)**  
   Reformulates coordinate regression into distribution-based learning for continuous spatial reasoning

2. **Reinforcement Learning for Visual-Reality Alignment (RL-VRA)**  
   Introduces geometric reward signals to ensure visual-reality alignment and spatial accuracy

3. **Reinforcement Learning from Aesthetic Feedback (RLAF)**  
   Employs learned aesthetic rewards to generate coherent and diverse compositions

### PosterCopilot Dataset

- **160K posters** with **2.6M layers** (1.2M text + 1.4M image/decorative elements)
- Spans **40+ distinct domains** from commercial promotions to public announcements
- Novel OCR-based pipeline addresses over-segmentation challenges in multi-layer datasets

---

## 📋 To-Do List

We are committed to making outstanding contributions to both academia and the graphic design industry with PosterCopilot. Our open-source plan includes:

### ✅ Released
- [x] Project page and documentation
- [x] Demo video

### 🚧 Coming Soon

- [ ] **Data Pipeline**
  - OCR-based layer merging pipeline
  - Dataset preprocessing scripts
  - Quality filtering tools

- [ ] **Test Dataset**
  - Curated evaluation benchmarks
  - Diverse domain coverage
  - Annotation files and metadata

- [ ] **Training Code**
  - Three-stage training framework implementation
  - PSFT training scripts
  - RL-VRA training scripts
  - RLAF training scripts
  - Configuration files and hyperparameters

- [ ] **Model Weights**
  - Pre-trained model checkpoints
  - Stage-wise model weights
  - Inference scripts and examples

- [ ] **Complete Documentation**
  - Installation guide
  - Training tutorial
  - Inference examples
  - API reference

---

## 📝 Citation

If you find PosterCopilot useful for your research, please consider citing:

```bibtex
@article{postercopilot2024,
  title={PosterCopilot: Toward Layout Reasoning and Controllable Editing for Professional Graphic Design},
  author={Wei, Jiazhe and Li, Ken and Lao, Tianyu and Wang, Haofan and Wang, Liang and Shan, Caifeng and Si, Chenyang},
  year={2024}
}
```

---

## 🔗 Links

- **Project Page**: [Coming Soon]
- **Paper**: [Coming Soon]
- **Demo**: [Coming Soon]

---

## 📧 Contact

For questions and collaborations, please contact:
- Jiazhe Wei: [jzw6545@gmail.com](mailto:jzw6545@gmail.com)
- Chenyang Si: [chenyangsi@smail.nju.edu.cn](mailto:chenyangsi@smail.nju.edu.cn)

---

## 📄 License

This project is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).

---

## 🙏 Acknowledgments

We thank all contributors and the research community for their valuable feedback and support.
