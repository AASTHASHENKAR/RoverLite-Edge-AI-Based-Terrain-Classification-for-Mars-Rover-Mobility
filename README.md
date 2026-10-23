# RoverLite-Edge-AI-Based-Terrain-Classification-for-Mars-Rover-Mobility
# RoverLite: Edge AI-Based Terrain Classification for Mars Rover Mobility

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/roverlite-mars-terrain-classifier/blob/main/RoverLite_Colab_Training_Notebook.ipynb)

A compact **MobileNetV2 transfer-learning model** that classifies a Mars terrain image patch as **Soil, Bedrock, Sand or Big Rock**. It reports accuracy and macro-F1 on held-out images, measures CPU inference latency and model size, and exports a TensorFlow Lite model for edge deployment.

> Deep Learning (Elective) mini project.

## Why Edge AI?
Radio signals between Earth and Mars take minutes, so a rover cannot wait for instructions from the ground to avoid soft sand. A small model that runs on the device itself, with no cloud, is the natural fit.

## Dataset
[AI4Mars](https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars) (NASA/JPL), loaded through the [Hugging Face mirror](https://huggingface.co/datasets/hassanjbara/AI4MARS). It provides rover images with pixel-level terrain masks (`0` soil, `1` bedrock, `2` sand, `3` big rock, `255` ignore).

Because the labels are segmentation masks, the notebook converts them into a classification task:
1. Resize each image and mask, then split them into four quadrant patches.
2. Keep a patch only if enough valid mask pixels exist and one class clearly dominates.
3. Label the patch with its dominant class.
4. Split train / validation / test **by source image**, so patches from the same photo never leak across splits.

## Model and method
- Backbone: MobileNetV2 pretrained on ImageNet (frozen), input 224x224
- Head: Global Average Pooling, Dropout (0.2), Dense softmax (4 classes)
- Class weights to handle imbalance; EarlyStopping and ReduceLROnPlateau
- Evaluation: accuracy, macro-F1, classification report, confusion matrix
- Edge step: Keras model size, CPU latency per patch, TensorFlow Lite dynamic-range export
- Demo rule: a predicted **Sand** or confidence below 0.60 triggers an advisory caution (a prototype rule, not a rover safety system)

## How to run
1. Click the **Open in Colab** badge above (or upload the notebook to Colab).
2. Runtime > Change runtime type > **T4 GPU** (CPU also works, but is slower).
3. Runtime > **Run all**. The dataset cell streams images from Hugging Face and may take several minutes.

To run locally: `pip install -r requirements.txt` and open the notebook in Jupyter.

## Results


| Metric | Value |
|--------|-------|
| Test accuracy (%) | -0.8080 (80.80%) |
| Test macro-F1 | 0.6301 |
| Keras model size (MB) |  9.24 MB |
| TFLite model size (MB) | 250.70 MB |
| Latency is measured on this Colab runtime; it is not a Raspberry Pi result. |

## Limitations
- Patch labels are the dominant mask class, which simplifies complex scenes.
- Labels can be incomplete or imbalanced.
- Latency is measured on a Colab CPU, not on a Raspberry Pi or a real rover.
- This is a learning prototype, not a validated safety system.

## References
1. NASA Open Data, [AI4Mars: A Dataset for Terrain-Aware Autonomous Driving on Mars](https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars)
2. Hugging Face, [AI4MARS dataset mirror](https://huggingface.co/datasets/hassanjbara/AI4MARS)
3. Swan et al., *AI4MARS: A Dataset for Terrain-Aware Autonomous Driving on Mars*, CVPR Workshops 2021
4. NASA/JPL, [SPOC terrain classification](https://nasa-jpl.github.io/SPOC/)

## Author
Aastha Pravin Shenkar | K.J Somaiya College of Engineering
