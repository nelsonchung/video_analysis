# Reproducible synthetic Golden Dataset

Generate small CPU regression inputs (no large binary recordings are committed):

```sh
python scripts/generate_golden.py --output golden_dataset/generated
videoqa batch --manifest golden_dataset/generated/manifest.yaml --output reports/golden --suspect warn --unverifiable warn
```

The deterministic 64×48, 48-frame, 30 FPS BGR24 fixtures cover normal moving texture,
freeze, timestamp gaps (five missing capture intervals), green blocks, blur,
red color shift and alternating temporal echoes. CSV timestamps are synthetic ground
truth, not recovered capture timestamps. A generic profile emits quality candidates;
the color case includes a synthetic neutral RGB reference and emits a color-distance
candidate. This Euclidean RGB comparison does not establish real color accuracy or
ΔE00 compliance, which require a calibrated chart and color conversion.

Generate the complete planned four-format / three-resolution smoke matrix:

```sh
python scripts/generate_golden.py --output golden_dataset/matrix --matrix
videoqa batch --manifest golden_dataset/matrix/manifest.yaml --output reports/matrix --suspect warn --unverifiable warn
```

The matrix uses three frames per input to verify decoding dimensions and frame counts;
it does not establish quality sensitivity or FPS stability. FFmpeg must provide
`libx264`, `libx265` and `mjpeg` encoders. RAW inputs use tightly packed 8-bit NV12;
the compressed inputs are elementary H.264, HEVC and concatenated MJPEG streams.

These fixtures are a reproducible software baseline. Actual V9 samples, RAW layout
contracts, capture timestamps and calibrated scene thresholds remain necessary for
hardware acceptance. Do not use synthetic thresholds as production color or blur limits.
