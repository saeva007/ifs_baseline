# Q1000 lineage audit

## Bottom line

The current result does **not** prove that Pangu represents moisture extremes
better than numerical models. It shows only that, in the downloaded analysis,
Pangu has a slightly less-negative marginal Q1000 tail bias against ERA5. All
three forecasts still substantially underestimate the ERA5 tail.

The present comparison is not publication-ready because forecast lead,
12-step window duration, ERA5 Q provenance, finite sample membership, and
below-ground 1000-hPa values are not all controlled.

## Confirmed code findings

1. `run_pangu_onnx_2025.py` uses the official Pangu order
   `Z,Q,T,U,V`, official 13 pressure levels, and the 24 h ONNX model. Q1000 is
   read from model channel Q at 1000 hPa. This part is consistent with the
   official implementation.
2. Commit `c1e1065` introduced a `lead12_23h` input-directory convention but no
   tracked generator for that product. Commit `9bdac99` later correctly
   documented the available ONNX path as lead24h and Tianji/IFS as
   `12 <= lead < 24 h`. Commit `958008a` then changed the production runbook,
   submitter, and audit expectation back to `lead12_23h` without adding a
   12--23 h Pangu inference path. This is a provenance regression.
3. The old station interpolator discarded `init_time` and lead metadata. The
   downstream audit therefore trusted a filename rather than actual forecast
   lead.
4. The old 12-step builder did not verify hourly continuity. With the default
   Pangu initialization hours `00/12`, 12 adjacent records could span about
   5.5 days while Tianji/IFS windows span 12 h.
5. ERA5 Q1000/Q925 were reconstructed from pressure-level T and RH when native
   specific-humidity station files were absent. A Q-quality reference analysis
   should use native ERA5 Q.
6. The downloaded old table is not a strict all-source finite complete case:
   Tianji/IFS use 1,753,004 rows, whereas Pangu uses 1,749,163 rows.
7. Tianji Q1000 and DP1000 are independent fields in
   `merged_final_all_vars.nc`; Pangu DP1000 is derived from Q1000. DP1000 is
   therefore a consistency view, not independent evidence.
8. The upstream script that creates Tianji `merged_final_all_vars.nc` is not in
   the tracked repositories. Its raw forecast-cycle stitching and Q/DP source
   metadata still require inspection on the server.

## Why Pangu can look less weak

- Pangu is initialized from ERA5 and evaluated against ERA5 reference analysis,
  so it shares an analysis family with the reference.
- The original local ONNX path is lead24h, whereas the updated canonical
  station product is the separately documented 12--23 h stitched product.
  They must not be mixed under one Pangu label.
- The apparent Pangu advantage is concentrated mainly in June--September,
  where marginal Q is strongly controlled by the warm/moist seasonal state.
- Positive-weight IDW interpolation cannot create a value above all neighboring
  grid values, so station interpolation is unlikely to explain a stronger tail.
- AI smoothing is a documented tendency, not a rule that every variable and
  every lead must have a lower marginal quantile than every NWP product.

For the downloaded source-full table, Pangu Q1000 MAE is 2.107 g kg-1 versus
2.153 for IFS and 2.214 for Tianji; P95 biases are -2.410, -2.890, and
-3.461 g kg-1, respectively. These rankings must be treated as diagnostic only
until the lineage audit passes.

## Implemented safeguards

- Pangu station interpolation now preserves per-time `init_time` and
  `forecast_lead_hours` and checks their consistency.
- Bulk Pangu inference defaults to all 24 initialization hours so 12 adjacent
  valid times can form an actual 12 h sequence.
- Tianji, IFS, ERA5, and Pangu dataset builders reject non-hourly axes.
- Pangu-2025 builds require an explicit expected lead range and either
  per-time metadata or the narrowly scoped documented 00/12 UTC stitched schedule.
- ERA5 q-core builds require native Q1000/Q925 provenance.
- `audit_pangu_tianji_q1000_lineage.py` reports raw-grid, station, and model
  dataset lineage plus common-finite-sample elevation sensitivity.

## Updated canonical-station product

The active updated data-check path now uses
`pangu_station_2025_lead12_23h_canonical.nc`. The corrected product is first
verified against the legacy station file and the canonical Tianji station
coordinates. For this known hourly stitched product, leads 12--23 h are
reconstructed from the documented 00/12 UTC cycle schedule when per-time lead
metadata are absent. `submit_corrected_pangu_q1000_checks.sh` rebuilds only the
Pangu source-full dataset and then runs the Q1000 lineage and mechanism checks.

## Required decision test

Use `q1000_paired_complete_case_elevation_metrics.csv` and require the same
conclusion in at least the low-elevation (`<100 m`) subset. If the Pangu tail
advantage disappears there, the old ranking was influenced by below-ground
1000-hPa extrapolation or sample composition. If it remains after lead matching,
native ERA5 Q, and date-block uncertainty, it is a real variable-specific result
and should be reported as “less underestimation,” not “stronger extremes.”

## References

- Pangu-Weather official model layout: https://github.com/198808xc/Pangu-Weather
- Bi et al. (2023), including smoothness and extreme-quantile limitations:
  https://doi.org/10.1038/s41586-023-06185-3
- ERA5 pressure-level construction and reference-analysis context:
  https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation
