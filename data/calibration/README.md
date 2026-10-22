# PCA-PQ calibration inputs

These inputs are used by the output-path-weighted PCA-PQ row of the main comparison.
The text is the fixed WikiText-2 training stream. The coordinate-weight arrays
contain the original fitting weights for BF16 E2B (35 × 256) and E4B (42 × 256).

Each weight is the mean squared calibration gate multiplied by the squared norm
of its output-projection column. The weights are normalized to mean one, floored
at 0.02, and normalized again. The accompanying metadata gives checkpoint hashes,
array shapes, and calibration settings.

The BF16 E2B and E4B fits use their corresponding arrays. The W4A16 E4B fit reuses
the original E4B array. The W4A16 E2B fit derives its weights from the QAT-parent
checkpoint with `scripts/derive_pca_pq_weights_e2b.py`.

The text and arrays retain their original bytes. Metadata paths point to this
package. Dataset attribution and terms are in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
