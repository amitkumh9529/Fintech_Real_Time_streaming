# Evaluation report status

The evaluation report has not yet been regenerated for `data/creditcard_2023.csv`.
The previous report was based on a different, time-stamped dataset and its
metrics and amount totals do not apply to the current input.

After feature preparation, training, and champion selection, run:

```powershell
python ml_pipeline/evaluate.py
```

The generated report will evaluate the untouched stratified test partition and
will label `Amount` totals in dataset units rather than assuming a currency or
claiming real-world losses prevented.
