# Third-party assets

- **Original method and client partitions:** *Aggregation Delayed Federated Learning*, Ye Xue, Diego Klabjan and Yuan Luo ([paper](https://arxiv.org/abs/2108.07433)). The supplied RADFed partitions are used unchanged. Their download locations and measured archive checksums are in the README. The original datasets retain their own terms; this repository does not grant a new license for those partitions.
- **Character vocabulary helper:** `language_utils.py`, including its archived copy, supplies the vocabulary and encoding used by the reused Shakespeare data format. It was preserved from the supplied RADFed working source; no separate license file accompanied that source. Attribution to the original method is retained above.
- **MobileNetV2 and bundled checkpoint:** torchvision's `mobilenet_v2-7ebf99e0.pth`, downloaded from the official PyTorch model distribution. SHA-256: `7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f`. Architecture and package documentation: [torchvision MobileNetV2](https://docs.pytorch.org/vision/stable/models/mobilenetv2.html). Package terms: [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE). The checkpoint is a separate pretrained asset; no additional license grant for it is made here.
- **Dependencies:** PyTorch, torchvision, NumPy and Pillow remain subject to their distribution licenses. Exact versions are in `DEPENDENCIES.md` and `requirements-pytorch.txt`.

No repository-wide license is inferred from a paper citation or from a dependency's license.
