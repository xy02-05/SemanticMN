<div align="center">

# Semantic Anchoring for Robotic Action Representations

**Yuan Xu\*<sup>1</sup> · Youheng Shi\*<sup>1</sup> · Chengyang Li<sup>2,3</sup> · Wentao Zhu<sup>2</sup> · Yizhou Wang<sup>1</sup>**

<sup>1</sup>Peking University &nbsp; <sup>2</sup>Eastern Institute of Technology, Ningbo &nbsp; <sup>3</sup>Shanghai Jiao Tong University

\* Equal contribution

[![Project Page](https://img.shields.io/badge/Project%20Page-SemanticMN-blue?style=flat-square&logo=googlechrome&logoColor=white)](https://xy02-05.github.io/SemanticMN/)
[![Paper](https://img.shields.io/badge/arXiv-2607.13597-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.13597)
[![Video](https://img.shields.io/badge/Video-YouTube-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=oIIPenJ80wg)

</div>

---

![teaser](https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/figures/teaser.png)

> **TL;DR:** VLA fine-tuning on limited robot data erodes its inherited semantic structure and undermines generalization. Inspired by mirror neuron theory, we probe this erosion and reveal its correlation with model performance. We introduce a plug-and-play semantic alignment method that consistently improves performance on simulation benchmarks and real-robot tasks.

---

## Demos

### In-Distribution Tasks

<table align="center">
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/gifs/grape_pick_place_success.gif" width="180" height="135"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/gifs/cup_stacking_success.gif" width="180" height="135"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/gifs/box_packing_success.gif" width="180" height="135"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/gifs/cabinet_storage_success.gif" width="180" height="135"/></td>
  </tr>
  <tr>
    <td align="center"><em>pick up the grapes and<br/>place it on the plate</em></td>
    <td align="center"><em>stack the cups</em></td>
    <td align="center"><em>place the toy bear into the box<br/>and close both side flaps</em></td>
    <td align="center"><em>place the sponge in the cabinet<br/>and close the door</em></td>
  </tr>
</table>

### Out-of-Distribution Generalization

<table align="center">
  <tr>
    <td align="center"><img src="https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/gifs/ood_language_success.gif" width="180" height="135"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/gifs/ood_object_success.gif" width="180" height="135"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/gifs/ood_visual_success.gif" width="180" height="135"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/gifs/ood_position_success.gif" width="180" height="135"/></td>
    <td align="center"><img src="https://raw.githubusercontent.com/xy02-05/SemanticMN/main/assets/gifs/ood_task_success.gif" width="180" height="135"/></td>
  </tr>
  <tr>
    <td align="center">Language Variation</td>
    <td align="center">Novel Object</td>
    <td align="center">Visual Distraction</td>
    <td align="center">Position Variation</td>
    <td align="center">Compositional Tasks</td>
  </tr>
</table>

---

## Supported Models & Environments

| Category | Supported |
|----------|-----------|
| VLA Models | [`pi0`](https://github.com/Physical-Intelligence/openpi), [`SpatialVLA`](https://github.com/SpatialVLA/SpatialVLA) |
| Simulators | [`LIBERO`](https://github.com/Lifelong-Robot-Learning/LIBERO), [`SimplerEnv`](https://github.com/simpler-env/SimplerEnv) |
| Datasets | BridgeData V2 (RLDS), LIBERO (LeRobot) |

---

## Citation

```bibtex
@misc{xu2026semanticanchoringroboticaction,
      title={Semantic Anchoring for Robotic Action Representations},
      author={Yuan Xu and Youheng Shi and Chengyang Li and Wentao Zhu and Yizhou Wang},
      year={2026},
      eprint={2607.13597},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.13597},
}
```

---

## Acknowledgement

This project builds on [openpi](https://github.com/Physical-Intelligence/openpi), [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA), and [StarVLA](https://github.com/starVLA/starVLA). We sincerely appreciate the work their authors have shared with the community.
