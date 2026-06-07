# Isolated Stage Power Summary

Method:
- `dsp-only.csv` and `nn-only.csv` were captured as separate 60 s PPK2 runs.
- The board was powered from PPK2 as the only supply source during measurement.
- Stage attribution uses baseline-subtracted average current rather than same-run UART phase alignment.

Results at `3.6 V`:
- DSP-only mean current: `20.505 mA`
- DSP-only baseline current: `17.570 mA`
- DSP-only dynamic current: `2.935 mA`
- DSP-only dynamic power: `10.566 mW`
- NN-only mean current: `20.898 mA`
- NN-only baseline current: `17.523 mA`
- NN-only dynamic current: `3.375 mA`
- NN-only dynamic power: `12.151 mW`

Dynamic stage split:
- DSP share: `46.51%`
- NN share: `53.49%`
- NN exceeds DSP by `0.440 mA` dynamic current (`1.585 mW`)

Source traces:
- `benchmarks/traces/dsp-only.csv`
- `benchmarks/traces/nn-only.csv`

The full CSVs are retained outside Git due to size. Checksums and derived trace visuals are tracked in:
- `benchmarks/results/power_trace_manifest.json`
- `docs/assets/ppk2-isolated-stage-traces.png`
