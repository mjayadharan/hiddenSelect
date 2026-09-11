# R### — reproduce

## Environment

```
python  <version>   (env: <path>)
<key packages and versions>
pdflatex (TeX Live <year>)
```

Machine notes: joblib's `loky` backend fails under the sandbox here — use
`backend="multiprocessing"`; never `threading` with SciPy LSODA. Set
`MPLCONFIGDIR` to a writable path for the figure stage.

## Ordered regeneration

```bash
cd analysis
OMP_NUM_THREADS=1 python run_stage1.py --workers 8
OMP_NUM_THREADS=1 python run_stage2.py --workers 8
python make_joins.py
python make_tables.py
cd ../figures && MPLCONFIGDIR=$TMPDIR python make_figures.py
cd ../analysis && python verify_<id>.py           # gates
cd .. && pdflatex -interaction=nonstopmode -halt-on-error report.tex   # twice
```

Wall clock on the reference machine (<N> workers): <per-stage timings>.

## Determinism

Deterministic given the seed list <seeds>; the only RNG is <what>
(`RNG_SEED = <value>`). Gate G1 asserts <the inherited-stage equality>.
