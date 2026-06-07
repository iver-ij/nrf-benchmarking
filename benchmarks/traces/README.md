# PPK2 Raw Traces

Full PPK2 CSV exports are not tracked in Git because each retained 60 s capture is about 166 MB.

Tracked records derived from those captures:

- `benchmarks/results/power_trace_manifest.json` records file sizes, sample counts, means, durations, and SHA-256 hashes.
- `docs/assets/ppk2-matched-runtime-traces.png` visualizes the `AI-only` and `AI + WiFi` traces.
- `docs/assets/ppk2-isolated-stage-traces.png` visualizes the `DSP-only` and `NN-only` traces.
- `benchmarks/results/power_comparison_summary.md` and `benchmarks/results/isolated_stage_power_summary.md` contain the retained numeric summaries.

To regenerate the checked-in summaries from local raw exports:

```bash
python benchmarks/generate_power_summaries.py --trace-dir /path/to/ppk2/csv_exports
```

Expected raw export names:

- `ai-only.csv`
- `ai+wifi.csv`
- `dsp-only.csv`
- `nn-only.csv`
