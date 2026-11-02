# Experiment Artifact Manifest

This directory mixes final results, supporting analyses, and retired negative
experiments. Do not infer paper status from file presence alone.

## Final

- `exp_4tier_s*.csv` - final capacity-ladder quality-gate runs.
- `presentation_summary.csv` - PHM Korea 2026 slide table values, rounded from
  the 8-seed `exp_4tier_s*.csv` repeated-measurement mean.
- `shares_confirmed.csv` - PHM Korea 2026 routing-distribution table.
- `final_models/cnn_base_s*.pt` - base CNN checkpoints.
- `final_models/cnn_x300_s*.pt` - x300 CNN checkpoints.
- `final_models/cnn_large_s*.pt` - large CNN checkpoints.
- `final_models/noise_est_7.6k.pt` - legacy compact estimator checkpoint used
  by existing `exp_4tier_s*.csv` artifacts.
- `fig_scripts/make_pareto_params_capacity.py` - final cost/F1 Pareto figure.
- `fig_scripts/make_capacity_method_figs.py` and `fig_scripts/make_capacity_figs.py` - capacity robustness figures.

## Supporting

- `imp_noiseaug*.csv` - noise-augmentation checks.
- `imp_prune*.csv`, `exp_structured_s*.csv`, `exp_quality_prune_s*.csv` - compression and pruning studies.
- `oracle_audit.csv` - oracle headroom audit.
- `exp_bn_*.csv` / scripts - BatchNorm evaluation artifact checks.
- `fig_scripts/make_*` - presentation figure generation scripts.

## Retracted Or Negative

- `exp_qweight_route_s*.csv` - superseded qweight routing experiments; early apparent wins were affected by BatchNorm eval mistakes.
- `exp_qw20_route_s*.csv`, `q20route_s*.csv`, `qsparsity_s*.csv` - routing variants that do not dominate the best single compressed/deployed point.
- `experts_*`, `route_*`, `gatemain_*`, `hardgate_*` - specialist or gate variants retained for audit, not the final claim.
- MOMENT/SNRGate artifacts outside this directory are legacy audit material; see `REPRODUCIBILITY_FINDINGS.md`.

## Canonical Command

```bash
python experiments/quality_routing/run.py --seed 42
```
