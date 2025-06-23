# Knowledge-Driven Imitation Learning: Enabling Generalization Across Diverse Conditions

[[Paper]]() [[Project Page]]()

**Authors**: Zhuochen Miao, Jun Lv, Hongjie Fang, Yang Jin, Cewu Lu

![pipeline](assets/pipeline.png)

## Getting Started

### Requirement

To set up the environment, run:

```bash
conda env create -f environment.yaml
conda activate Knowledge
```

For real-world evaluation, you also need to run:

```bash
pip install pyrealsense2 pynput
```

and place `flexiv_rdk/` into the root folder of this project.

### Knowledge Template

1. Capture an RGBD image of the target object.
2. Save the files `color.png`, `depth.png`, and `intr.npy` into the directory `data/templates/[object_name]`.

To manually select keypoints, run:

```bash
python -m scripts.select --template_path data/templates/[object_name]
```

The selected keypoints will be saved as `points.npy` in the same directory.

Currently, templates for **mug** and **coaster** are provided.

To prepare a dataset (e.g., `data/dataset/mug`), modify `config.yaml` as follows:

```yaml
anchor_name: "anchors.npy"
num_anchors: 12 # Total number of keypoints
objects:
  - "data/templates/coaster"
  - "data/templates/mug"
average:
  - []
  - [0, 1, 2, 3] # For the mug rim, DINOv2 features may vary slightly; average the first four keypoints to reduce this variation.
calib:
  relative_path: "calib/0212"
  camera_serial:
    - "135122075425"
    - "104422070042"
```

To perform template matching on all training data, run:

```bash
python -m scripts.anchor --dataset_path data/dataset/mug
```

Add `--vis` to visualize the generated anchors.

### Training

To train the model, run:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --data_path data/dataset/mug \
  --num_action 20 \
  --ckpt_dir ./logs/mug \
  --batch_size 60 \
  --num_epochs 1000 \
  --save_epochs 100 \
  --num_workers 24 \
  --seed 233 \
  --policy_type anchor \
  --aug
```

It may takes several hours.

### Realworld Evaluation

To evaluate the trained model on real-world data, run:

```bash
python eval.py \
  --ckpt logs/mug/policy_epoch_1000_seed_233.ckpt \
  --calib data/calib/eval \
  --policy_type anchor \
  --cfg data/dataset/mug/config.yaml \
  --vis
```

## Acknowledgement

- Our codebase is adapt from [RISE](https://github.com/rise-policy/rise), realeased under CC BY-NC-SA 4.0 License.
- The diffusion module is from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), released under MIT License.
- The feature extrctor [Dinov2](https://github.com/facebookresearch/dinov2) is provided under the Apache License 2.0.
- We also refer to [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy), [P3PO](https://github.com/mlevy2525/P3PO) for parts of the implementation.

## Citation

```bibtex

```
