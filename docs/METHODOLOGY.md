# Methodology notes

This document records what can be stated from the public research summary and assets. The full implementation and labeled dataset are restricted; details not specified there are called out rather than inferred.

## Input representation

For each participant, the input is a symmetric, whole-brain 200 × 200 Pearson correlation matrix. Its strict upper triangle contains

\[
\binom{200}{2} = \frac{200 \cdot 199}{2} = 19{,}900
\]

continuous edge values. The public CSV has one additional `participant_id` field, so it has 19,901 columns total. In the labeled study, the binary target is `ADHD_Outcome` (1 = ADHD; 0 = control); the public sample contains no target column.

## Signed-triangle criterion

For a closed triangle with edge weights \(W_{AB}\), \(W_{BC}\), and \(W_{CA}\), the reported signed criterion flags a triangle when

\[
W_{AB} W_{BC} W_{CA} < 0.
\]

This is a property of the signed graph. It is not, on its own, a measure of a biological process. The public materials do not specify the exact subject-level aggregation and normalization used to form every triadic feature.

## Descriptor families and implementation boundary

- **Kinetic flow deceleration:** described as a shortest-path/resistance-style feature, with Dijkstra on inverted correlation weights given only as an example. The exact edge-to-distance transform, treatment of negative/zero weights, and aggregation are not specified publicly.
- **Dead-end crash / directional asymmetry:** described as strong positive inbound connectivity paired with weak or negative outbound connectivity. The directed representation and exact calculation from the symmetric input matrix are not specified publicly.
- **Inhibitory mass / antagonism:** described as cumulative negative-edge magnitude around nodes or subsystems. Exact aggregation and normalization are not specified publicly.

These details matter for mathematical reproduction. No particular transform, threshold, or formula should be inferred from the conceptual figures.

## Validation and stacking

The reported evaluation uses 5 outer folds and 5 inner folds. KS feature selection is fitted within training folds. The regularized linear meta-estimator produces `LINEAR_META_RISK_SCORE` out-of-fold within the inner training process; Optuna tunes the LightGBM stage. Each outer held-out fold is used for evaluation, with reported ROC-AUC values listed in the main [README](../README.md#zero-leakage-5x5-nested-cross-validation).

The public summary does not provide fold assignments, the exact KS statistic/selection rule, search space, random seeds, or the definition and aggregation procedure for OOF Information Gain. Accordingly, the reported metrics and rankings can be presented, but cannot be independently recomputed from the public sample.
