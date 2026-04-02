# Structure from Motion

**Full Structure-from-Motion pipeline implemented from scratch** — feature detection and matching, essential matrix estimation, camera pose recovery, triangulation, and GTSAM factor graph bundle adjustment — producing sparse and dense 3D reconstructions from a custom multi-view image dataset of a Buddha statue. Achieves a mean reprojection error of **3 pixels** after bundle adjustment.

---

## Demo

![SfM Demo](assets/demo.gif)

> Left: feature matches between image pairs (LoFTR). Right: recovered sparse point cloud with camera poses (frustums). Final output: dense reconstruction of Buddha statue.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Stage 1 — Feature Detection and Matching](#stage-1--feature-detection-and-matching)
  - [SIFT](#sift-scale-invariant-feature-transform)
  - [LoFTR](#loftr-detector-free-local-feature-matching)
  - [SIFT vs LoFTR Comparison](#sift-vs-loftr-comparison)
- [Stage 2 — Geometric Estimation](#stage-2--geometric-estimation)
  - [Fundamental Matrix](#fundamental-matrix)
  - [Essential Matrix](#essential-matrix)
  - [RANSAC Robust Estimation](#ransac-robust-estimation)
  - [Pose Recovery](#pose-recovery)
- [Stage 3 — Triangulation](#stage-3--triangulation)
- [Stage 4 — Bundle Adjustment (GTSAM)](#stage-4--bundle-adjustment-gtsam)
  - [Factor Graph Formulation](#factor-graph-formulation)
  - [Prior Factors](#prior-factors)
  - [Optimization](#optimization)
- [Stage 5 — Dense Reconstruction](#stage-5--dense-reconstruction)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

Structure from Motion recovers the 3D structure of a scene and the camera poses that observed it from a set of 2D images alone — no depth sensor, no IMU, no GPS. It is the foundational algorithm behind photogrammetry, autonomous mapping, and visual SLAM.

This project implements the full incremental SfM pipeline from scratch:

```
Multi-view images
    → Feature detection + matching (SIFT / LoFTR)
    → Essential matrix estimation (8-point + RANSAC)
    → Camera pose recovery (R, t from E)
    → Triangulation (DLT linear method)
    → Incremental reconstruction (register new views, triangulate new points)
    → Bundle adjustment (GTSAM factor graph with prior factors)
    → Dense reconstruction (multi-view stereo)
    → Sparse + dense point cloud output
```

The key differentiator is the **GTSAM factor graph bundle adjustment** — rather than using a black-box optimizer, the scene is modeled as a probabilistic factor graph where camera poses and 3D points are variables connected by reprojection factors and prior factors, and optimized via Levenberg-Marquardt.

---

## Pipeline Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Input Images (Buddha)                      │
│              N multi-view RGB images, known K                 │
└──────────────────────────┬────────────────────────────────────┘
                           │
             ┌─────────────▼──────────────┐
             │   Feature Detection        │
             │   SIFT keypoints +         │
             │   descriptors              │
             │      OR                    │
             │   LoFTR dense matches      │
             │   (transformer-based)      │
             └─────────────┬──────────────┘
                           │ Putative matches
             ┌─────────────▼──────────────┐
             │   Geometric Verification   │
             │   Fundamental matrix F     │
             │   Essential matrix E       │
             │   RANSAC outlier rejection │
             └─────────────┬──────────────┘
                           │ Inlier matches + E
             ┌─────────────▼──────────────┐
             │   Pose Recovery            │
             │   SVD decomposition of E   │
             │   4-solution disambiguation│
             │   R, t per camera pair     │
             └─────────────┬──────────────┘
                           │ Camera poses (initial)
             ┌─────────────▼──────────────┐
             │   Triangulation (DLT)      │
             │   2D correspondences →     │
             │   3D point cloud           │
             └─────────────┬──────────────┘
                           │ Noisy 3D points + poses
             ┌─────────────▼──────────────┐
             │   Bundle Adjustment        │
             │   GTSAM Factor Graph       │
             │   Prior factors on poses   │
             │   Reprojection factors     │
             │   Levenberg-Marquardt opt  │
             └─────────────┬──────────────┘
                           │ Refined poses + points
             ┌─────────────▼──────────────┐
             │   Dense Reconstruction     │
             │   Multi-view stereo        │
             │   depth fusion             │
             └─────────────┬──────────────┘
                           │
             ┌─────────────▼──────────────┐
             │   Output                   │
             │   Sparse point cloud       │
             │   Dense point cloud        │
             │   Camera poses (frustums)  │
             └────────────────────────────┘
```

---

## Stage 1 — Feature Detection and Matching

### SIFT: Scale-Invariant Feature Transform

SIFT detects keypoints that are stable under scale, rotation, and moderate illumination change by building a **scale-space representation** of the image and finding extrema across scales.

**Scale space construction:**

```
L(x, y, σ) = G(x, y, σ) * I(x, y)

where G(x, y, σ) = (1/2πσ²) exp(-(x²+y²)/2σ²)
```

The image is convolved with Gaussians at progressively larger σ, organized into octaves (each octave halves the image resolution). **Difference of Gaussians (DoG)** approximates the Laplacian of Gaussian:

```
D(x, y, σ) = L(x, y, kσ) − L(x, y, σ)
```

Keypoints are detected at local extrema of D across scale and space — a point must be larger or smaller than all 26 neighbors (8 spatial + 9 above + 9 below in scale space).

**Descriptor construction:**

Around each keypoint, a 16×16 pixel neighborhood is divided into 4×4 cells. In each cell, an 8-bin gradient orientation histogram is computed. The 4×4×8 = **128-dimensional descriptor vector** is L2-normalized for illumination invariance.

**Matching:** Lowe's ratio test — a match between descriptors `d1` and `d2` is accepted only if:

```
dist(d1, d2_nearest) / dist(d1, d2_second_nearest) < 0.75
```

This rejects ambiguous matches where two database descriptors are similarly close to the query — a primary source of false matches.

```python
sift = cv2.SIFT_create(nfeatures=5000)
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# FLANN matcher with ratio test
matcher = cv2.FlannBasedMatcher()
matches = matcher.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]
```

**Limitation on the Buddha dataset:** The statue has large textureless regions (smooth stone surfaces) and repetitive fine detail (carved patterns). SIFT produces sparse matches on smooth surfaces and ambiguous matches on repetitive texture — motivating the switch to LoFTR.

---

### LoFTR: Detector-Free Local Feature Matching

LoFTR (Sun et al., 2021) replaces the detect-then-describe paradigm with a **detector-free** approach: it directly computes dense matches between image pairs using a transformer that attends over the full image feature maps of both images simultaneously.

**Architecture:**

```
Image A                    Image B
    │                          │
    ▼                          ▼
CNN backbone (FPN)         CNN backbone (FPN)
    │                          │
    └──────────┬───────────────┘
               ▼
    Flatten → 1D feature sequences
    + positional encoding
               │
               ▼
    ┌──────────────────────┐
    │  Self-Attention      │  ← each image attends to itself
    │  Cross-Attention     │  ← each image attends to the other
    │  × L transformer     │
    │    layers            │
    └──────────┬───────────┘
               │
               ▼
    Coarse match predictions
    (feature map resolution)
               │
               ▼
    Fine-level refinement
    (sub-pixel localization)
               │
               ▼
    Dense correspondences (x₁,y₁) ↔ (x₂,y₂)
```

**Why LoFTR outperforms SIFT on the Buddha dataset:**

- **Textureless regions:** SIFT finds no keypoints on smooth stone surfaces. LoFTR's cross-attention allows it to match based on global context — e.g., a smooth region near a carved edge is matched by its spatial relationship to surrounding features, not local gradient structure.
- **Repetitive patterns:** SIFT's ratio test rejects ambiguous matches on repetitive carved patterns. LoFTR resolves ambiguity via global image context — the transformer knows *which* instance of a repeated pattern it is looking at based on its position relative to the whole scene.
- **Result:** Significantly more matches on the Buddha statue, better coverage of smooth and curved surfaces, lower reprojection error after BA.

```python
from loftr import LoFTR, default_cfg

matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
matcher.eval().cuda()

data = {
    'image0': img0_tensor,   # 1×1×H×W grayscale
    'image1': img1_tensor,
}
with torch.no_grad():
    matcher(data)

mkpts0 = data['mkpts0_f'].cpu().numpy()   # matched keypoints in image 0
mkpts1 = data['mkpts1_f'].cpu().numpy()   # matched keypoints in image 1
mconf  = data['mconf'].cpu().numpy()      # match confidence scores
```

---

### SIFT vs LoFTR Comparison

| Property | SIFT | LoFTR |
|---|---|---|
| Match density | Sparse (keypoint-only) | Dense (all image regions) |
| Textureless surfaces | Poor — no keypoints detected | Good — global context used |
| Repetitive patterns | Ambiguous — ratio test rejects | Resolved via cross-attention |
| Inference speed | Fast (CPU) | Slower (GPU, transformer) |
| Implementation | From scratch (OpenCV) | Pretrained model |
| Matches on Buddha dataset | Few, clustered at edges | Many, full surface coverage |

---

## Stage 2 — Geometric Estimation

### Fundamental Matrix

The fundamental matrix F encodes the epipolar constraint between two uncalibrated views:

```
x'ᵀ F x = 0

for any correspondence (x, x') in homogeneous coordinates
```

F is a 3×3 rank-2 matrix with 7 degrees of freedom. It is estimated using the **normalized 8-point algorithm**:

**Normalization:** Scale and translate point coordinates so they have zero mean and RMS distance √2 from origin. This conditions the linear system, dramatically improving numerical stability:

```python
def normalize_points(pts):
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    scale = np.sqrt(2) / np.mean(np.linalg.norm(pts_c, axis=1))
    T = np.array([
        [scale,  0,    -scale * centroid[0]],
        [0,      scale, -scale * centroid[1]],
        [0,      0,      1]
    ])
    pts_norm = (T @ np.hstack([pts, np.ones((len(pts),1))]).T).T
    return pts_norm[:, :2], T
```

**8-point algorithm:** Each correspondence gives one linear constraint on the 9 elements of F. With ≥8 correspondences, solve `Af = 0` via SVD:

```python
def estimate_fundamental(pts1, pts2):
    pts1_n, T1 = normalize_points(pts1)
    pts2_n, T2 = normalize_points(pts2)

    # Build constraint matrix A (N×9)
    A = []
    for (x1,y1), (x2,y2) in zip(pts1_n, pts2_n):
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    A = np.array(A)

    # Solve via SVD: f = last column of V
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # Denormalize
    F = T2.T @ F @ T1
    return F / F[2, 2]
```

---

### Essential Matrix

Given calibrated cameras (known intrinsic matrix K), the essential matrix E encodes the same epipolar geometry but in normalized image coordinates:

```
E = Kᵀ F K

x̂'ᵀ E x̂ = 0    where x̂ = K⁻¹ x  (normalized coordinates)
```

E has 5 degrees of freedom (3 for rotation, 2 for translation direction — scale is unrecoverable from images alone) and satisfies the additional constraint:

```
2 E Eᵀ E − trace(E Eᵀ) E = 0
```

which means its two non-zero singular values must be equal. This is enforced after estimation by averaging the two non-zero singular values from SVD.

---

### RANSAC Robust Estimation

Raw feature matches contain outliers — false matches that satisfy the ratio test but don't correspond to the same physical point. Fitting F or E directly to outlier-contaminated data produces a degenerate result. **RANSAC** (Random Sample Consensus) robustly estimates the model by:

```
for i in range(max_iterations):
    sample ← draw 8 random correspondences
    F_candidate ← estimate_fundamental(sample)
    inliers ← {(x,x') : |x'ᵀ F x| / normalization < threshold}
    if |inliers| > best_inlier_count:
        best_F = F_candidate
        best_inliers = inliers

F_final ← refit on best_inliers
```

**Number of iterations** to guarantee finding an outlier-free sample with probability p=0.99:

```
N = log(1 - p) / log(1 - (1 - ε)^s)

where:
  ε = outlier ratio (estimated from data)
  s = sample size (8 for fundamental matrix)
```

For 50% outlier ratio: N ≈ 1177 iterations. For 30% outliers: N ≈ 235 iterations.

**Sampson distance** is used as the inlier criterion (first-order approximation to reprojection error, cheaper than full triangulation):

```
d_sampson = (x'ᵀ F x)² / ((Fx)₀² + (Fx)₁² + (Fᵀx')₀² + (Fᵀx')₁²)
```

---

### Pose Recovery

The essential matrix decomposes into four (R, t) solutions via SVD:

```python
def recover_pose(E, pts1_n, pts2_n):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

    # Four candidate solutions
    R1, R2 = U @ W @ Vt, U @ W.T @ Vt
    t1, t2 = U[:, 2], -U[:, 2]
    candidates = [(R1,t1),(R1,t2),(R2,t1),(R2,t2)]

    # Cheirality check: correct solution has most points
    # in front of both cameras (positive depth)
    best_R, best_t, best_count = None, None, 0
    for R, t in candidates:
        P1 = np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = np.hstack([R, t.reshape(3,1)])
        pts3d = triangulate_points(P1, P2, pts1_n, pts2_n)
        # Count points with positive depth in both cameras
        depth1 = pts3d[:, 2]
        depth2 = (R @ pts3d.T + t.reshape(3,1))[2]
        count = np.sum((depth1 > 0) & (depth2 > 0))
        if count > best_count:
            best_R, best_t, best_count = R, t, count

    return best_R, best_t
```

The **cheirality check** selects the unique correct solution: points must lie in front of (positive depth from) both cameras. Three of the four solutions place points behind at least one camera.

---

## Stage 3 — Triangulation

Given matched 2D points `x₁, x₂` and camera projection matrices `P₁, P₂`, triangulation recovers the 3D point `X` by solving the overdetermined linear system via DLT (Direct Linear Transform):

Each correspondence gives two equations:

```
x₁ × (P₁X) = 0   →   [ x₁(P₁³ᵀ) − P₁¹ᵀ ] X = 0
                       [ y₁(P₁³ᵀ) − P₁²ᵀ ] X = 0
```

Stacking both views gives a 4×4 system `AX = 0`, solved via SVD (solution = last row of V):

```python
def triangulate_point(P1, P2, x1, y1, x2, y2):
    A = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]   # convert from homogeneous
```

**Limitation of linear triangulation:** DLT minimizes an algebraic error, not the geometrically meaningful reprojection error. For high-accuracy reconstruction, triangulated points are used as initialization for the non-linear bundle adjustment.

---

## Stage 4 — Bundle Adjustment (GTSAM)

Bundle adjustment jointly optimizes all camera poses and 3D point positions to minimize the total reprojection error across all observations — it is the global refinement step that turns noisy incremental estimates into an accurate reconstruction.

### Factor Graph Formulation

The scene is modeled as a **factor graph** — a bipartite graph connecting variable nodes (camera poses, 3D points) to factor nodes (measurement constraints):

```
Variables:
  X_i ∈ SE(3)   — camera pose i (rotation + translation)
  L_j ∈ R³      — 3D landmark j

Factors:
  Prior factor on X_0         — anchors the coordinate frame
  Prior factor on X_1         — fixes scale (translation direction only)
  GenericProjectionFactor     — reprojection constraint per (camera, point) observation

Factor graph:

  [Prior]─── X_0 ───[Proj]──── L_1 ────[Proj]─── X_1 ───[Prior]
                        \                  /
                    [Proj]── L_2 ──[Proj]
                        \                  /
                    [Proj]── L_3 ──[Proj]
                              ...
```

The **joint probability** of the factor graph is:

```
P(X, L | Z) ∝ ∏_i prior(X_i) × ∏_{i,j} exp(-½ ||z_{ij} − π(X_i, L_j)||²_Σ)

where:
  z_{ij}        = observed 2D keypoint of landmark j in camera i
  π(X_i, L_j)  = projection of L_j through camera i
  Σ             = measurement noise covariance (isotropic, σ=1px)
```

**Maximizing this posterior = minimizing the sum of squared reprojection errors** (MAP estimation under Gaussian noise = least squares).

---

### Prior Factors

Prior factors impose soft constraints on specific variables, anchoring the optimization to prevent gauge freedom (the reconstruction is defined only up to a similarity transform — without priors, the optimizer can rotate, translate, and scale the whole scene arbitrarily).

```python
import gtsam

graph = gtsam.NonlinearFactorGraph()
initial = gtsam.Values()

# Noise models
prior_noise_pose  = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.1, 0.1, 0.1,    # rotation (rad)
              0.3, 0.3, 0.3])   # translation (m)
)
measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # 1px isotropic

# Prior on first camera pose — fixes coordinate frame origin
graph.add(gtsam.PriorFactorPose3(
    gtsam.symbol('x', 0),
    initial_pose_0,
    prior_noise_pose
))

# Prior on second camera pose — fixes scale
graph.add(gtsam.PriorFactorPose3(
    gtsam.symbol('x', 1),
    initial_pose_1,
    prior_noise_pose
))
```

**Why two prior factors:** A single prior on pose 0 fixes position and orientation but leaves scale free — the optimizer can shrink or expand the entire reconstruction. The prior on pose 1 constrains the inter-camera translation magnitude, fixing scale. Together they remove all 7 degrees of gauge freedom (3 rotation + 3 translation + 1 scale).

### Reprojection Factors

Each 2D observation of landmark `j` in camera `i` adds a `GenericProjectionFactor`:

```python
K = gtsam.Cal3_S2(fx, fy, 0, cx, cy)  # camera intrinsics

for cam_idx, lm_idx, observed_pt in observations:
    graph.add(gtsam.GenericProjectionFactorCal3_S2(
        gtsam.Point2(*observed_pt),
        measurement_noise,
        gtsam.symbol('x', cam_idx),
        gtsam.symbol('l', lm_idx),
        K
    ))
```

### Optimization

```python
# Initialize with triangulated poses and points
for i, pose in enumerate(initial_poses):
    initial.insert(gtsam.symbol('x', i), pose)
for j, point in enumerate(triangulated_points):
    initial.insert(gtsam.symbol('l', j), gtsam.Point3(*point))

# Levenberg-Marquardt optimization
params = gtsam.LevenbergMarquardtParams()
params.setMaxIterations(100)
params.setRelativeErrorTol(1e-5)

optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
result = optimizer.optimize()

# Extract refined poses and points
refined_poses  = [result.atPose3(gtsam.symbol('x', i)) for i in range(n_cams)]
refined_points = [result.atPoint3(gtsam.symbol('l', j)) for j in range(n_points)]
```

**Levenberg-Marquardt** interpolates between Gauss-Newton (fast convergence near minimum) and gradient descent (stable far from minimum) by scaling the update step:

```
(JᵀJ + λI) Δx = -Jᵀr

λ small → Gauss-Newton (fast, aggressive)
λ large → gradient descent (slow, stable)
λ adapts per iteration based on whether the step improved the cost
```

---

## Stage 5 — Dense Reconstruction

After bundle adjustment produces accurate camera poses, **multi-view stereo** densifies the sparse point cloud by computing per-pixel depth maps and fusing them:

1. **Depth map per camera:** For each pixel in camera `i`, search along the epipolar line in neighboring cameras for the best photometric match (NCC patch similarity)
2. **Depth fusion:** Project all depth maps into a shared 3D volume, filter inconsistent depths (geometric consistency check across views), and merge into a unified dense point cloud

---

## Dataset

Custom multi-view image set of a **Buddha statue**, captured under controlled indoor lighting:

- ~30 images captured in a 360° arc around the statue
- Overlapping coverage between consecutive frames (~60% overlap)
- Challenges: smooth stone surfaces (low texture), repetitive carved patterns, specular highlights on polished regions
- Camera: [camera model], focal length approximately known, calibration matrix K estimated via checkerboard calibration

---

## Results

| Metric | SIFT | LoFTR |
|---|---|---|
| Feature matches per pair | Low — sparse on smooth surfaces | High — dense across full surface |
| Reprojection error (pre-BA) | ~8–12 px | ~5–7 px |
| Reprojection error (post-BA) | — | **~3 px** |
| Point cloud coverage | Partial — edges only | Full surface coverage |

**Mean reprojection error of 3 pixels** after GTSAM bundle adjustment confirms accurate recovery of both camera poses and 3D structure.

### Qualitative Results

- Camera poses recovered as frustums, arranged in the correct 360° arc around the statue
- Sparse point cloud captures the overall shape of the Buddha — head, shoulders, carved details
- Dense reconstruction fills in smooth surface regions missed by SIFT-only sparse reconstruction
- LoFTR matches on smooth stone surfaces are the primary driver of dense coverage quality

> *(Add point cloud visualization, camera frustum plot, and reprojection overlay images here)*

---

## Installation

```bash
git clone https://github.com/sarthak-talwadkar/Structure-from-Motion.git
cd Structure-from-Motion
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, NumPy, OpenCV, PyTorch (LoFTR), GTSAM Python bindings, Open3D, Matplotlib

**Install GTSAM Python bindings:**
```bash
pip install gtsam
```

**Install LoFTR:**
```bash
pip install kornia
# Download pretrained weights
wget https://drive.google.com/file/d/outdoor_ds.ckpt -O weights/outdoor_ds.ckpt
```

---

## Usage

### Run full SfM pipeline

```bash
python run_sfm.py \
    --images data/buddha/ \
    --K data/calibration.txt \
    --matcher loftr \
    --output results/
```

### Run with SIFT (comparison)

```bash
python run_sfm.py \
    --images data/buddha/ \
    --K data/calibration.txt \
    --matcher sift \
    --output results_sift/
```

### Bundle adjustment only (on existing reconstruction)

```bash
python bundle_adjust.py \
    --poses results/poses.npy \
    --points results/points3d.npy \
    --observations results/observations.npy \
    --K data/calibration.txt \
    --output results/refined/
```

### Visualize reconstruction

```bash
python visualize.py \
    --points results/refined/points3d.npy \
    --poses results/refined/poses.npy
```

---

## Project Structure

```
Structure-from-Motion/
├── features/
│   ├── sift.py             # SIFT detection, description, ratio-test matching
│   └── loftr_matcher.py    # LoFTR dense matching wrapper
├── geometry/
│   ├── fundamental.py      # Normalized 8-point algorithm
│   ├── essential.py        # E from F, rank-2 enforcement
│   ├── ransac.py           # RANSAC with Sampson distance inlier criterion
│   ├── pose_recovery.py    # SVD decomposition of E, cheirality check
│   └── triangulation.py    # DLT linear triangulation
├── bundle_adjustment/
│   └── gtsam_ba.py         # GTSAM factor graph BA (prior + projection factors)
├── dense/
│   └── mvs.py              # Multi-view stereo depth estimation + fusion
├── utils/
│   ├── normalize.py        # Point normalization for 8-point algorithm
│   └── visualize.py        # Point cloud + camera frustum visualization
├── data/
│   ├── buddha/             # Input images
│   └── calibration.txt     # Camera intrinsic matrix K
├── run_sfm.py              # Full pipeline entry point
├── bundle_adjust.py        # Standalone BA on existing reconstruction
├── requirements.txt
└── README.md
```

---

## References

- Hartley, R. & Zisserman, A. *"Multiple View Geometry in Computer Vision."* Cambridge University Press, 2004. *(Fundamental/Essential matrix, DLT triangulation)*
- Lowe, D. *"Distinctive Image Features from Scale-Invariant Keypoints."* IJCV 2004. [[Paper]](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94) *(SIFT)*
- Sun, J. et al. *"LoFTR: Detector-Free Local Feature Matching with Transformers."* CVPR 2021. [[Paper]](https://arxiv.org/abs/2104.00680)
- Dellaert, F. & GTSAM Contributors. *"GTSAM."* [[Repo]](https://github.com/borglab/gtsam)
- Fischler, M. & Bolles, R. *"Random Sample Consensus."* CACM 1981. *(RANSAC)*

---

## Author

**Sarthak Talwadkar**
MS Robotics, Northeastern University — Autonomous Field Robotics Course
[LinkedIn](https://linkedin.com/in/sarthak-talwadkar) · [GitHub](https://github.com/sarthak-talwadkar)
