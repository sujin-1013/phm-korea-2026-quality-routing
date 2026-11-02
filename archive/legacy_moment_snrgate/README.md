# Legacy MOMENT/LoRA SNRGate Audit

The old MOMENT-S/B/L + LoRA SNRGate pipeline is retained in its historical
source locations for auditability, but it is not the current public method.

Relevant legacy paths:

- `src/phm_routing/models/moment_wrapper.py`
- `src/phm_routing/models/lora.py`
- `src/phm_routing/models/router.py`
- `src/phm_routing/training/train_routing.py`
- `src/phm_routing/training/train_baseline.py`
- `experiments/exp00_sanity/`
- `experiments/exp02_routing/`
- `experiments/exp03_snrgate/`
- `experiments/exp_baselines/`

Install optional dependencies only when auditing this retired path:

```bash
pip install -e ".[legacy-moment,dev]"
```

`REPRODUCIBILITY_FINDINGS.md` records the reproduction failure of the old 0.583
headline and explains why the final repository now points to the CNN
capacity-ladder quality-routing path.
