# Power Comparison Summary

Matched 60 s PPK2 captures at 3.6 V:

- `AI-only` trace: `benchmarks/traces/ai-only.csv`
- `AI + WiFi` trace: `benchmarks/traces/ai+wifi.csv`
- Raw CSVs are retained outside Git due to size; checksums and derived trace visuals are tracked in `benchmarks/results/power_trace_manifest.json`.

Whole-run comparison:

- AI-only mean current: `21.606 mA`
- AI + WiFi mean current: `22.675 mA`
- WiFi overhead: `+1.069 mA` (`+4.95%`)
- WiFi overhead at 3.6 V: `+3.848 mW`

Event-level detection:

- AI-only: `34` steady-inference events, mean event energy `11,415.32 uJ`, mean event duration `107.99 ms`
- AI + WiFi: `32` steady-inference events, mean event energy `11,662.52 uJ`, mean event duration `108.08 ms`

Notes:

- Summaries were generated with the auto-fallback `steady_inference` detection profile in `benchmarks/analyze_power.py`.
- UART timing logs were used to validate expected inference timing, but per-phase DSP/NN energy attribution is intentionally suppressed for these captures because the thresholded power windows are shorter than the full DSP+NN pipeline.
- Trace visualization: `docs/assets/ppk2-matched-runtime-traces.png`
