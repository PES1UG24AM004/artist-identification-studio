# ML Model Comparison Report: Validation Set Benchmark

**Evaluation Protocol:**
- **Split:** Validation Set (`data/processed/dataset_manifest.csv`)
- **Sample Size:** 200 paintings (exactly 20 per artist class across 10 balanced classes)
- **Models Evaluated:**
  1. **SVM Baseline:** HSV Color Hist + Hu Moments + GLCM (559 dims) -> StandardScaler -> PCA(100) -> RBF SVC
  2. **ResNet-18 CNN (TTA OFF):** Pretrained ResNet-18 fine-tuned linear head, deterministic single-crop (Resize 256 -> CenterCrop 224)
  3. **ResNet-18 CNN (TTA ON):** Same fine-tuned ResNet-18 with Horizontal-Flip TTA (softmax probability averaging of original and mirrored views)

## 1. Overall Model Comparison Table

| Model | TTA | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Mean Latency (ms/image) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM Baseline | OFF | 53.50% | 0.5405 | 0.5350 | 0.5330 | 0.5330 | 16.63 ms |
| ResNet-18 CNN (TTA OFF) | OFF | 81.00% | 0.8212 | 0.8100 | 0.8084 | 0.8084 | 44.61 ms |
| ResNet-18 CNN (TTA ON) | ON | 81.50% | 0.8301 | 0.8150 | 0.8131 | 0.8131 | 79.96 ms |

## 2. Test-Time Augmentation (TTA) Impact Analysis

- **Total predictions changed:** 6 / 200 (3.0%)
- **Previously incorrect predictions corrected:** 2
- **Previously correct predictions degraded:** 1
- **Net correct prediction gain:** +1
- **Accuracy difference:** +0.50 percentage points (81.0% -> 81.5%)
- **Macro F1 difference:** +0.0047 (0.8084 -> 0.8131)

## 3. Per-Class Artist Performance

| Artist | SVM F1 | SVM Correct | CNN F1 (TTA OFF) | CNN F1 (TTA ON) | F1 Diff | Correct (TTA OFF) | Correct (TTA ON) | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Albrecht Durer | 0.9048 | 19/20 | 0.8837 | 0.9091 | +0.0254 | 19/20 | 20/20 | 20 |
| Alfred Sisley | 0.8095 | 17/20 | 0.8182 | 0.8182 | +0.0000 | 18/20 | 18/20 | 20 |
| Edgar Degas | 0.2857 | 6/20 | 0.8780 | 0.8780 | +0.0000 | 18/20 | 18/20 | 20 |
| Francisco Goya | 0.5000 | 9/20 | 0.7647 | 0.7879 | +0.0232 | 13/20 | 13/20 | 20 |
| Pablo Picasso | 0.4242 | 7/20 | 0.6842 | 0.7027 | +0.0185 | 13/20 | 13/20 | 20 |
| Paul Gauguin | 0.6667 | 13/20 | 0.8235 | 0.8235 | +0.0000 | 14/20 | 14/20 | 20 |
| Pierre-Auguste Renoir | 0.5000 | 9/20 | 0.8837 | 0.8636 | -0.0201 | 19/20 | 19/20 | 20 |
| Rembrandt | 0.4091 | 9/20 | 0.8500 | 0.8205 | -0.0295 | 17/20 | 16/20 | 20 |
| Titian | 0.3415 | 7/20 | 0.8000 | 0.8293 | +0.0293 | 16/20 | 17/20 | 20 |
| Vincent van Gogh | 0.4889 | 11/20 | 0.6977 | 0.6977 | +0.0000 | 15/20 | 15/20 | 20 |

## 4. Key Findings

1. **Deep Learning Superiority Over Classical Features:** The ResNet-18 architecture outperforms the hand-crafted feature SVM baseline by **+27.50% accuracy** (81.00% vs 53.50%) and **+0.2753 Macro F1** (0.8084 vs 0.5330). Hand-crafted color and texture descriptors struggle with spatial composition and painterly technique variations.
2. **Horizontal-Flip TTA Effect:** Incorporating horizontal-flip probability averaging alters 6 predictions out of 200. It successfully corrects 2 paintings while misclassifying 1, yielding a net gain of +1 correctly classified artwork. Overall accuracy slightly increases from 81.00% to 81.50% (+0.50 pp) and Macro F1 from 0.8084 to 0.8131 (+0.0047).
3. **Class-Specific Sensitivities:**
   - **Albrecht Dürer:** TTA achieved perfect recall (20/20, 100%), improving F1 from 0.8837 to 0.9091.
   - **Francisco Goya & Pablo Picasso:** Showed precision and F1 improvements under TTA due to orientation invariance in expressive brushwork.
   - **Pierre-Auguste Renoir & Rembrandt:** Suffered slight degradations (-0.0201 and -0.0295 F1 respectively), indicating directional lighting or compositional asymmetry can be sensitive to lateral inversion.
4. **Latency Trade-off:** Horizontal-Flip TTA doubles the forward pass evaluations, increasing mean per-image latency from 44.61 ms to 79.96 ms on CPU. Both remain well within interactive real-time application constraints (<100 ms).

## 5. Limitations & Protocol Notes

- **Held-Out Test Set Preservation:** This evaluation was strictly performed on the 200 validation images to protect the held-out test split (200 images) against data leakage and hypothesis contamination.
- **Sample Size:** With 20 samples per class in the validation split, single-sample prediction changes represent a 5.0% shift in per-class recall and a 0.5% shift in overall accuracy.
- **TTA Invariance Assumption:** Horizontal flipping assumes artist style is laterally symmetric; while effective on average, artists with strong directional chiaroscuro or composition signatures can occasionally experience degraded confidence.
