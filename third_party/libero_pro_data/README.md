---
license: cc-by-4.0
task_categories:
- robotics
tags:
- vision-language-action
- vla
- benchmark
- robot-learning
---

# Dataset Card for LIBERO-PRO Perturbation Dataset

[**Project Page**](https://zxy-mllab.github.io/LIBERO-PRO-Webpage/) | [**Paper**](https://huggingface.co/papers/2510.03827) | [**Code**](https://github.com/Zxy-MLlab/LIBERO-PRO)

This dataset contains the **`bddl`** and **`init`** files of LIBERO-PRO configurations under **object**, **relation**, **semantic**, **task**, and **environment** perturbations. The dataset supports direct integration with the [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO) framework to evaluate Vision-Language-Action (VLA) models beyond rote memorization.

---

## Dataset Details

### Dataset Description

- **Curated by:** LIBERO-PRO Research Team  
- **Affiliation:** MLLab, Huazhong University of Science and Technology  
- **Language(s) (NLP):** English (instructional text)  
- **License:** CC-BY-4.0 (for dataset artifacts), MIT (for codebase)
- **Primary Purpose:** Evaluation of VLA models under structured perturbations  

This dataset extends the original [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO/) by introducing **systematic perturbations** in five dimensions:
1. **Object Perturbation:** Modifies object appearance, color, and scale to test adaptability to visual shifts.  
2. **Position Perturbation:** Relocates objects within feasible spatial bounds to evaluate the model’s adaptability to spatial position changes.  
3. **Semantic Perturbation:** Paraphrases natural language commands to probe linguistic robustness.  
4. **Task Perturbation:** Redefines task logic and target states to test procedural generalization.
5. **Environment Perturbation:** Replaces working environments to evaluate cross-environment robustness.  

Each perturbation includes corresponding **`init` files** (initial environment configurations) and **`bddl` files** (behavioral descriptions in BDDL format).

---

## Seven-Case Robustness Extension

The repository also includes a 40-task evaluation set covering seven
BDDL-configured robustness cases. Each category contains
10 tasks from each of `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`.

| Folder | Evaluation case |
| --- | --- |
| `01_visual_noise_glare` | Lighting and observation noise |
| `02_camera_view_angle` | Camera position and orientation |
| `03_runtime_object_move` | Runtime target-object movement |
| `04_object_texture` | Object appearance and texture |
| `05_view_occlusion` | View occlusion by scene objects |
| `06_object_shape` | Target-object shape scaling |
| `07_initial_pose_position_angle` | Initial position and yaw changes |

The 280 BDDL files use this layout:

```text
bddl_files/<category>/bddl/<suite>/<task>.bddl
```

Shared original initialization states are stored under:

```text
init_files/<suite>/<task>.pruned_init
```

Where available, the corresponding `.init` files are included as well. The
runtime object movement case uses a near-grasp trigger with a maximum
end-effector-to-target distance of `0.09 m` and a step-160 fallback.

The `metadata/` directory contains a portable dataset index, task-specific
perturbation manifest, and the latest static validation report. File checksums
are listed in `SHA256SUMS.txt`.

The custom `:perturbation_config` fields require the LIBERO-Pro-aware parser
and evaluation integration from the project codebase.

---

## Uses

**How to use:**
1. Copy all **`.bddl`** files to:
   ```bash
   LIBERO-PRO/libero/libero/bddl_files/
   ```
2. Copy all **`init`** files to:
   ```bash
   LIBERO-PRO/libero/libero/init_files/
   ```
3. Follow the quick start instructions provided in the [LIBERO-PRO README](https://github.com/Zxy-MLlab/LIBERO-PRO#readme).

---

## Dataset Structure

Each perturbation category contains:
- **`init/`**: Environment initialization files defining object placement and world state.  
- **`bddl/`**: Task goal definitions in Behavior Domain Definition Language.  

---

## Citation

If you use this dataset, please cite both the original LIBERO benchmark and the LIBERO-PRO project:

**BibTeX:**
```bibtex
@article{zhou2025liberopro,
  title={LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization},
  author={Xueyang Zhou and Yangming Xu and Guiyao Tie and Yongchao Chen and Guowen Zhang and Duanfeng Chu and Pan Zhou and Lichao Sun},
  journal={arXiv preprint arXiv:2510.03827},
  year={2025}
}

@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

---

## Dataset Card Authors

- Xueyang Zhou
- Yangming Xu

---

## Dataset Card Contact

For questions or issues, please contact:  
📧 **d202480819@hust.edu.cn**
