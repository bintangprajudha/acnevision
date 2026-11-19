# Benchmark of prior acne lesion detection results on ACNE04 / ACNE04-v2

Research date: 2026-11-13. User reference point (for comparison only, not a source): YOLOv8l @ 960px, 1 class, ACNE04-v2 circles converted to boxes, own split 963/120/121, test AP@50 0.562, mAP@50-95 ~0.20, P 0.611, R 0.557, F1 0.583 (SAHI).

Verification legend: **[full text read]** = numbers read from the paper's own PDF/HTML; **[abstract/listing only]** = numbers from an abstract, proceedings listing or search snippet, full text not read.

## Q1. Which papers report detection metrics on ACNE04 / ACNE04-v2?

### Takeaway
Only a handful of papers report box-level detection metrics on ACNE04. Each uses its own random split, crop scheme and metric (no official detection split exists). On the original ACNE04 labels, peer-reviewed or arXiv AP values are low (0.10–0.23). On ACNE04-v2, the only standard-metric detection result found is SAT-AcneDet (SPIE SHDH 2026): mAP@50 = 0.608. Its full text could not be read.

### Cited Findings

**Benchmark table (original ACNE04 = Wu 2019 boxes, 1,457 images, 1 class unless stated)**

| # | Paper (venue, year) | Model | Input / crop | Split | Classes | Metric (as defined) | Value | Verif. |
|---|---|---|---|---|---|---|---|---|
| 1 | Min, Lee, Lee, "ACNet: Mask-Aware Attention with Dynamic Context Enhancement for Robust Acne Detection" (arXiv 2105.14891, 2021; IEEE SMC 2021) | Faster R-CNN, ResNet101 | 640×640 crops, 320px overlap | random 80/20 of 1,457 imgs (18,983 boxes) | 1 | "mAP" (IoU threshold not stated) | 10.3 | full text, Table I |
| 1 | same | R-FCN, ResNet101 | same | same | 1 | mAP | 14.0 | Table I |
| 1 | same | Rashataprucksa et al. 2020, ResNet101 | same | same | 1 | mAP | 14.7 | Table I |
| 1 | same | **ACNet** (dual ResNet101) | same | same | 1 | mAP | **20.5** | Table I |
| 2 | Zhang J., Zhang L., et al., "Learning High-Quality Proposals for Acne Detection" (arXiv 2207.03674, 2022) | proposed (two-stage, high-quality proposals) | not stated for ACNE04 | random 80/20 = 1,165 train / 292 test | 1 | "AP and AR at IoU=0.5 with 100 detection boxes at most" | **0.22** (baselines copied: FRCNN 0.103, R-FCN 0.140, Rashataprucksa 0.147, ACNet 0.205) | full text (HTML) |
| 3 | Wei X., Zhang L., et al., "Towards Accurate Acne Detection via Decoupled Sequential Detection Head" (arXiv 2301.12219, 2023) | Mask R-CNN + DSDH, ResNet-101 | 640×640 crops | random 80/20 (counts not given) | 1 | "PASCAL-style AP", ≤100 dets/img | **23.0** (copied baselines R-FCN 14.0, Rashataprucksa 14.7, ACNet 20.5) | full text (HTML), Table 2 |
| 4 | Sangha & Rizvi, "Detection of acne by deep learning object detection" (medRxiv 10.1101/2021.12.05.21267310, 2021, preprint) | YOLOv5 (COCO-pretrained; "773 layers, 141.8M params" ≈ YOLOv5x) | not verified | not verified | 1 and 4 (severity levels) | mAP@0.5 | 37.97 (1-class); 26.50 (4-class) | abstract/snippet only (medRxiv returned 403) |
| 5 | Wen H., Yu W., et al., "Acne detection and severity evaluation with interpretable CNN models" (Technology and Health Care 30:S143–S153, 2022, DOI 10.3233/THC-228014) | several detectors incl. YOLO-type | n/a | n/a | n/a | mAP | 0.536 "on an open source dataset" (dataset not confirmed as ACNE04); counting MAE 3.49 on "filtered ACNE04" | snippet only |

Sources: ACNet Table I and setup: [ar5iv 2105.14891](https://ar5iv.labs.arxiv.org/html/2105.14891); Zhang 2022: [arXiv 2207.03674v1](https://arxiv.org/html/2207.03674v1); Wei 2023 DSDH: [arXiv 2301.12219](https://arxiv.org/html/2301.12219) / [abs](https://arxiv.org/abs/2301.12219); Sangha & Rizvi: [medRxiv](https://www.medrxiv.org/content/10.1101/2021.12.05.21267310v1); Wen 2022: [DOI 10.3233/THC-228014](https://dx.doi.org/10.3233/THC-228014).

**Benchmark table (ACNE04-v2 = Gazeau 2024 circle annotations)**

| # | Paper | Model | Input | Split / protocol | Metric | Value | Verif. |
|---|---|---|---|---|---|---|---|
| 6 | Gazeau et al., "AcneAI: A new acne severity assessment method using digital images and deep learning", MICCAI 2024, LNCS pp. 68–78, DOI 10.1007/978-3-031-72086-4_7 | AcneAI (UNet-EfficientNet segmentation + per-lesion scoring) | n/a | **ACNE04-v2 used only as an external test set** (not used in training) | TPR / FPR / Precision / "AP" / AUC (non-standard, see Q2) | TPR 0.52, FPR 0.09, Precision 0.52, "AP" 0.45, AUC 0.79 (threshold 1.81) | full text, Table 1 |
| 6 | same | YOLOv8 (Ultralytics) trained on AcneAI's **private** segmentation data converted to boxes | 1280×1280 | external test on ACNE04-v2 | same | TPR 0.54, FPR 0.14, Precision 0.54, "AP" 0.57, AUC 0.72 (thr 0.18); at thr 0.29: TPR 0.35, FPR 0.04, Precision 0.73 | full text, Table 1 |
| 6 | same | YOLOv8 trained **on ACNE04-v2**, tested on private clinical data | 1280 | cross-dataset | same | TPR 0.32, FPR 0.04, Precision 0.32, "AP" 0.26, AUC 0.85 (thr 0.16) | full text, Table 1 |
| 7 | Yiwen Ding & Guozhong Wang, "SAT-AcneDet" (YOLO11n + Gaussian Soft-SPD, Soft-SPPF, EMA-driven DynamicTAL, scale-aware dynamic hybrid loss), SPIE Proc. Vol. 14389, SHDH 2026 | SAT-AcneDet (YOLO11n-based) vs YOLO11n | not known | not known | Recall, mAP@50 | ACNE04-v2: Recall 0.518 → **0.601**, mAP@50 0.573 → **0.608**. AcneSCU: Recall 0.588 → 0.608, mAP@50 0.634 → 0.651 | abstract/listing only |

Sources: AcneAI PDF: [papers.miccai.org 2216_paper.pdf](https://papers.miccai.org/miccai-2024/paper/2216_paper.pdf); MICCAI page: [042-Paper2216](https://papers.miccai.org/miccai-2024/042-Paper2216.html); SAT-AcneDet: [SPIE Vol. 14389 listing](https://spie.org/Publications/Proceedings/Volume/14389). The SPIE page could not be fetched directly; the numbers come from the search engine's rendering of that page.

**Other relevant but non-detection or non-comparable works**
- Zhang & Ma, "Acne Detection by Ensemble Neural Networks", Sensors 22(18):6828, 2022: ACNE04 80/20 split. Reports counting "accuracy (number)" and RMSE only, no mAP/P/R. Values: proposed 84.60% / RMSE 2.74; F-RCNN 73.97% / 3.39; YOLOv3 63.70% / 3.37; Wu et al. 84.11% / 2.33. The text says YOLOv3 64.70%, which conflicts with its own table. It cites Rashataprucksa R-FCN mAP "up to 28.3%" — [PMC9505228](https://pmc.ncbi.nlm.nih.gov/articles/PMC9505228/)
- Wu et al. ICCV 2019 (ACNE04 origin) is a grading and counting (LDL) paper. Its comparisons are against classification and LDL baselines. I found no box mAP in it. The Cardiff PDF returned 403, so I could not read its tables. — [mlanthology](https://mlanthology.org/iccv/2019/wu2019iccv-joint/), [orca.cardiff.ac.uk PDF](https://orca.cardiff.ac.uk/125008/1/AcneLDL_ICCV2019.pdf)
- AcneFormer (Sensors 2026, DOI 10.3390/s26082533) uses ACNE04-v2 as an external set for "lesion agreement" in a **classification** model. No detection mAP was seen. It points to a Kaggle mirror at kaggle.com/datasets/karmagames/acne04-v2 — [MDPI](https://doi.org/10.3390/s26082533)
- The Wei 2023 ACNE-DET/AcneSCU results (276 images, 31,777 lesions, 10 classes; best Mask R-CNN+DSDH AP 45.6) are on a **different dataset** — [arXiv 2301.12219](https://arxiv.org/html/2301.12219)

### Inferences
- The "AP ≈ 0.10–0.23" family (ACNet, Zhang 2022, DSDH) shares one lineage (same group, Sichuan Univ./Xian Jiang for 2 and 3). Later papers copy baselines from ACNet's table rather than re-running them. Zhang 2022 defines its AP at IoU=0.5, but ACNet does not state an IoU. Treat these as roughly AP@0.5 on the original, noisier ACNE04 boxes, an 80/20 random split, and 640 crops.
- The user's AP@50 0.562 on ACNE04-v2 cannot be compared directly with these numbers. The original ACNE04 boxes contain about 18,983 lesions in 1,457 images. ACNE04-v2 has 32,443 circles in 1,204 images: the labels changed, and many missed lesions were added.

### Gaps
- Full text of SAT-AcneDet (split, input size, mAP@50-95, precision, F1, number of test images) was not accessible: spie.org blocked, and no arXiv version was found.
- Sangha & Rizvi and Wen 2022 full texts were not read (403 / paywall), so their split and input size are unverified.
- I did not find the Rashataprucksa et al. 2020 original (R-FCN, "up to 28.3%" mAP). Whether that value is on ACNE04 is unverified.

## Q2. What does the ACNE04-v2 paper (Gazeau et al., MICCAI 2024) report, and what is its protocol, split and repo?

### Takeaway
AcneAI does **not** provide a standard detection benchmark, train/test split or mAP on ACNE04-v2. ACNE04-v2 is used purely as an external test set. "AP" there is defined as correct predictions / total predictions, which is effectively precision and not COCO/VOC AP. The repo provides annotations only (no splits, no baselines, no leaderboard).

### Cited Findings
- Authors: Léa Gazeau, Hang Nguyen, Zung Nguyen, Mariia Lebedeva, Thanh Nguyen, Tat-Dat To, Jimmy Le Digabel, Jérome Filiol, Gwendal Josse, Clifford Perlis, Jonathan Wolfe (Torus AI / BelleTorus / Pierre Fabre). DOI 10.1007/978-3-031-72086-4_7. MICCAI page lists code as "N/A" — [MICCAI page](https://papers.miccai.org/miccai-2024/042-Paper2216.html)
- The authors write: "We evaluated our acne severity assessment system on the Acne04 dataset … which is not used in our training/validation." They criticise the original ACNE04 boxes: "not lesion-centered and/or are too big … sometimes more than one lesion are in the same bounding box; and lesions are sometimes missed". On the re-annotated set they report "AUC of 0.79. With a threshold of 1.81 we obtain a true positive rate of 0.52 and false positive rate of 0.09" (Sec. 3.2, p. 6–7) — [AcneAI PDF](https://papers.miccai.org/miccai-2024/paper/2216_paper.pdf)
- Sec. 4: "we trained several models using Ultralytics YOLOv8 (input size 1280x1280x3)". Table 1 caption: "First threshold is chosen by minimizing |FP − FN|, the second one is found by maximizing (TP − FP). Here AP stands for Average precision, and corresponds to the number of correct predictions over the total number of predictions." The numbers are in the Q1 table, p. 9 — [AcneAI PDF](https://papers.miccai.org/miccai-2024/paper/2216_paper.pdf)
- The paper does not state the IoU or matching criterion used to count a TP. A MICCAI reviewer also noted that "no information is provided on how the dataset was trained, if cross validation was used or division of the dataset" and asked the authors to explain the meaning of AP in Table 1 — [MICCAI reviews](https://papers.miccai.org/miccai-2024/042-Paper2216.html)
- Severity: ICC 0.81 vs ACNE04 levels (text; the abstract says 0.8) — [AcneAI PDF](https://papers.miccai.org/miccai-2024/paper/2216_paper.pdf)
- Repo `AIpourlapeau/acne04v2` has 1,204 images and 32,443 lesion circles (center + radius, COCO-like JSON), a `draw_acne_from_json.py` script and a `yolo_scripts/` folder. **No splits and no benchmark results.** License not stated — [GitHub](https://github.com/AIpourlapeau/acne04v2)
- The supplementary table reportedly lists ACNE04-v2 as 1,203 images / 32,376 lesions. This differs slightly from the repo's 1,204 / 32,443. It comes from a search snippet of the [MICCAI supplement](https://papers.miccai.org/miccai-2024/supp/2216_supp.pdf); I could not render the supplement's text (image-based PDF).

### Inferences
- The AcneAI YOLOv8 "AP 0.57" and "Precision 0.54" are **not** AP@50 and are not comparable to the user's AP@50 0.562. The model was trained on private data, tested on all of ACNE04-v2, and evaluated with an unspecified matching rule and a non-standard AP definition.
- Since no official split exists, every ACNE04-v2 detection paper (including the user's 963/120/121) defines its own split. Comparisons are only approximate unless the split is shared.

### Gaps
- The TP matching rule in AcneAI (IoU threshold or circle-centre distance) is not given in the main paper.
- I found no public leaderboard or Papers-with-Code task for ACNE04-v2 detection.

## Q3. Which result is SOTA, and is it truly comparable?

### Takeaway
- **ACNE04-v2:** the highest reported standard detection metric found is SAT-AcneDet, mAP@50 = 0.608 and Recall = 0.601 (YOLO11n baseline: 0.573 / 0.518), SPIE SHDH 2026. It is only partially comparable: its split, input size and confidence/IoU protocol are unknown.
- **Original ACNE04:** the best published (arXiv-level) value is Mask R-CNN + DSDH, AP = 23.0 (2023). It is not comparable to ACNE04-v2 numbers because the labels differ.

### Cited Findings
- SAT-AcneDet ACNE04-v2: mAP@50 0.573 → 0.608; Recall 0.518 → 0.601 — [SPIE Vol. 14389](https://spie.org/Publications/Proceedings/Volume/14389) (abstract/listing only)
- DSDH ACNE04: AP 23.0 vs ACNet 20.5 — [arXiv 2301.12219](https://arxiv.org/html/2301.12219)
- Zhang 2022 ACNE04: AP@0.5 0.22 — [arXiv 2207.03674](https://arxiv.org/html/2207.03674v1)

### Inferences
- The user's AP@50 0.562 (ACNE04-v2, own split, 960px, SAHI) sits just below SAT-AcneDet's YOLO11n baseline (0.573) and about 0.046 below SAT-AcneDet (0.608). The user's recall of 0.557 sits between the two SAT-AcneDet recall values (0.518 and 0.601). Recall depends on the confidence threshold, which SAT-AcneDet's listing does not give. Because the splits differ and SAT-AcneDet's protocol is unknown, differences of a few points may be within split-to-split variance. With about 121 test images, the user's own bootstrap CI is the right yardstick.
- The ACNE04-v2 results from AcneAI (TPR/precision ≈ 0.52–0.54) are on a different protocol and should not be used as the main benchmark row. They can be cited as "external-test precision/recall".
- To make a defensible claim, the user should (a) state the split explicitly and (b) ideally re-train a YOLO11n baseline at 640 on their own split, to calibrate against SAT-AcneDet's baseline of 0.573.

### Gaps
- No ACNE04-v2 paper reporting mAP@50-95 or F1 was found, so the user's mAP@50-95 ~0.20 and F1 0.583 have no published counterpart.
- 2025–2026 coverage may be incomplete. Google Scholar "cited by" pages for Wu 2019 and Gazeau 2024 were not accessible with these tools.

## Q4. Papers with suspicious numbers or different protocols

### Takeaway
Several high ACNE04 "mAP" figures (0.73–0.90) come from retracted, non-peer-reviewed, Roboflow-resplit or unverified sources. They should be excluded or footnoted.

### Cited Findings
- **"Utilizing Generative Adversarial Networks for Acne Dataset Generation in Dermatology"** (BioMedInformatics 4(2), 2024, DOI 10.3390/biomedinformatics4020059) is **RETRACTED**. It reported YOLOv8 mAP 73.6%, P 80.2%, R 65.3% using synthetic plus ACNE04 data. Exclude it — [MDPI DOI](https://doi.org/10.3390/biomedinformatics4020059)
- **"Automatic Acne Detection Model Based on Improved YOLOv7"** (IEEE Xplore doc 10810392, ~2024 conference) reportedly gives 83.7% mAP on ACNE04 (+4.57% over baseline). The full text was not accessible, and its split, IoU and class definitions are unverified. This value is roughly 4× all other ACNE04 results. It is likely a different protocol (e.g., Roboflow re-split, augmented test set, or AP at a loose IoU). Flag it as suspicious — [IEEE Xplore](https://ieeexplore.ieee.org/document/10810392/), [ResearchGate](https://www.researchgate.net/publication/387355075_Automatic_acne_detection_model_based_on_improved_YOLOv7)
- **Indonesian undergraduate thesis (UPI repository)**, "Identifikasi kulit wajah berjerawat menggunakan YOLOv8": P 88.4%, R 85.8%, F1 87%, mAP50 90% on ACNE04. It is not peer-reviewed, and its live test was much lower (P 78.31%, R 55.26%). Likely a Roboflow version — [repository.upi.edu/121635](https://repository.upi.edu/121635/)
- **Roboflow Universe mirrors** of ACNE04 (e.g., "acne04-detection", "DERMNETNZ + ACNE04", "acne04v2-stats-model") have their own re-splits and often augmented train/valid/test exports. Numbers trained on them are not comparable — [Roboflow acne04-detection](https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection), [Roboflow DERMNETNZ+ACNE04](https://universe.roboflow.com/muhammad-suzaki-zahran/dermnetnz---acne04/dataset/2), [Roboflow acne04v2-stats-model](https://universe.roboflow.com/john-hr2ni/acne04v2-stats-model)
- An independent GitHub project fine-tuning YOLOv8s on ACNE04 (original labels, 60 epochs) reports best val mAP@50 ≈ 0.325, with P and R around 0.3–0.4. This is a useful sanity anchor showing that original-label mAP@50 is low — [GitHub clearframe-acne-retouch](https://github.com/blackleg-sanji007/clearframe-acne-retouch) (snippet-level only)
- Multi-class or other-dataset results that are sometimes mis-cited as ACNE04: ACNE8M YOLOv8 mAP 0.69 (12 classes, own dataset) — [ResearchGate](https://www.researchgate.net/publication/389071791_ACNE8M_-_An_acnes_detection_and_differential_diagnosis_system_using_AI_technologies). AcneDet (Huynh et al., Diagnostics 2022) Faster R-CNN mAP 0.54 on its own smartphone dataset — [PMC9406819](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9406819/). ACNE-DET/AcneSCU (10 classes) — [arXiv 2301.12219](https://arxiv.org/html/2301.12219)
- Protocol caveats inside legitimate papers: ACNet, Zhang 2022 and DSDH all evaluate on 640 crops (or unspecified sizes), and ACNet does not state the IoU used for mAP — [ar5iv 2105.14891](https://ar5iv.labs.arxiv.org/html/2105.14891). The Sangha & Rizvi 4-class result uses **severity levels as box classes**, which is not a lesion-type taxonomy — [medRxiv](https://www.medrxiv.org/content/10.1101/2021.12.05.21267310v1)

### Inferences
- Any ACNE04 detection claim above about 0.6 mAP@50 on original labels should be treated as a protocol artefact until its split and evaluation are verified. Peer-reviewed or arXiv results on original labels top out around AP 0.23 (crop-level), and an independent YOLOv8s run reached about 0.33.

### Gaps
- I could not access the Improved YOLOv7 (IEEE 10810392) full text to confirm its protocol.
- I did not check whether the 2025 systematic review ("AI in the Assessment and Grading of Acne Vulgaris", ResearchGate 392445034) lists additional Transformer+CNN detection results on ACNE04/AcneSCU — [ResearchGate](https://www.researchgate.net/publication/392445034_Artificial_Intelligence_in_the_Assessment_and_Grading_of_Acne_Vulgaris_A_Systematic_Review)
