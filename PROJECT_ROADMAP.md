# Project Roadmap

This document outlines the milestone roadmap for the benchmark framework.

## Phase 1: Architecture
**Objective:** Establish the core abstractions and benchmark infrastructure.
**Deliverables:** Dataset Loader, XML Parser, Data Models, Prediction Models, MatchingEngine ABC, BenchmarkRunner, Metrics Skeleton, Dummy Engine.
**Expected output:** A runnable pipeline that executes a dummy matching engine.
**Dependencies:** None.
**Checklist:**
- [x] Dataset Loader
- [x] XML Parser
- [x] Data/Prediction Models
- [x] Engine & Runner Abstractions
- [x] Dummy Validation
**Completion status:** Completed.

## Phase 2: Feature Extraction
**Objective:** Implement the initial semantic feature extraction layer.
**Deliverables:** DINOv2 FeatureExtractor concrete class.
**Expected output:** Ability to extract global dense features from both Reference and Test images.
**Dependencies:** Phase 1, CoarseToFine abstractions.
**Checklist:**
- [ ] DINOv2 wrapper implementation
- [ ] Preprocessing for RGB and Thermal
**Completion status:** Pending.

## Phase 3: Global Similarity
**Objective:** Compute global similarity between reference features and test frame features.
**Deliverables:** Similarity computation logic (integrated with Feature Extractor).
**Expected output:** Cosine similarity maps/features.
**Dependencies:** Phase 2.
**Checklist:**
- [ ] Feature normalization
- [ ] Similarity metric implementation
**Completion status:** Pending.

## Phase 4: Coarse Localization
**Objective:** Identify candidate regions based on global similarity to narrow down the search space.
**Deliverables:** Heatmap Generation & Candidate Localizer.
**Expected output:** Coarse bounding boxes highlighting potential target locations.
**Dependencies:** Phase 3.
**Checklist:**
- [ ] Heatmap aggregation
- [ ] Thresholding and bounding box extraction
**Completion status:** Pending.

## Phase 5: Fine Matching
**Objective:** Extract local features and perform robust matching within the candidate region.
**Deliverables:** ALIKED Keypoint Detector and LightGlue Matcher.
**Expected output:** Sets of matched keypoints between the reference image and the candidate region.
**Dependencies:** Phase 4.
**Checklist:**
- [ ] ALIKED integration
- [ ] LightGlue integration
**Completion status:** Pending.

## Phase 6: Geometric Verification
**Objective:** Reject false positive matches and compute final transformations.
**Deliverables:** OpenCV RANSAC GeometricVerifier.
**Expected output:** Refined, verified bounding box for the target object.
**Dependencies:** Phase 5.
**Checklist:**
- [ ] GeometricVerifier interface finalized
- [ ] RANSAC logic
**Completion status:** In Progress (Interface only).

## Phase 7: Metrics
**Objective:** Accurately evaluate the performance of the matching pipeline against ground truth.
**Deliverables:** Full MetricsCalculator.
**Expected output:** Reports detailing IoU, Precision, Recall, and mAP.
**Dependencies:** Phase 6.
**Checklist:**
- [ ] IoU implementation
- [ ] Precision / Recall computation
- [ ] Aggregate reporting
**Completion status:** Pending.

## Phase 8: RGB + Thermal Validation
**Objective:** Ensure the unified pipeline handles both modalities flawlessly.
**Deliverables:** Unified evaluation script.
**Expected output:** Benchmark results showing performance on both RGB and Thermal datasets without pipeline changes.
**Dependencies:** Phase 7.
**Checklist:**
- [ ] Thermal preprocessing verification
- [ ] Cross-modal robustness testing
**Completion status:** Pending.

## Phase 9: Performance Optimization
**Objective:** Reduce latency and increase throughput for real-time constraints.
**Deliverables:** Optimized model weights (TensorRT, ONNX, INT8).
**Expected output:** Significant FPS increase during inference.
**Dependencies:** Phase 8.
**Checklist:**
- [ ] Export models to ONNX
- [ ] TensorRT calibration
**Completion status:** Pending.

## Phase 10: Jetson Deployment
**Objective:** Ensure the framework runs smoothly on edge hardware (NVIDIA Jetson).
**Deliverables:** Edge deployment scripts and environment configs.
**Expected output:** Benchmark runs successfully on target Jetson hardware.
**Dependencies:** Phase 9.
**Checklist:**
- [ ] JetPack compatibility validation
- [ ] Memory profiling
**Completion status:** Pending.

## Phase 11: Teknofest Submission
**Objective:** Finalize the project for competition integration.
**Deliverables:** TCP/UDP Client, JSON Output formatters, Live Camera integration.
**Expected output:** A real-time, deployable package ready for competition.
**Dependencies:** Phase 10.
**Checklist:**
- [ ] Network I/O module
- [ ] Final end-to-end testing
**Completion status:** Pending.
