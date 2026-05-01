# Two-Stage Anomaly Detection UI

This UI runs your two-stage pipeline:

- Stage 1: anomaly detection on encoded autoencoder features.
- Stage 2: attack-type classification for traffic predicted as anomaly.

## Run

From the project root:

```powershell
pip install -r UI/requirements.txt
streamlit run UI/app.py
```

## Input Data

- Default test source: `PREPROCESSING/processed_cicids2017.csv`
- You can also upload any CSV with compatible feature columns.

## Notes

- Stage-1 models are loaded from `MODELS/xgboost_model.pkl` and `MODELS/rf_model.pkl`.
- Stage-2 models are loaded from `MODELS/xgboost_stage2.pkl` and `MODELS/randomforest_stage2.pkl`.
- If stage-2 PKL files are not fitted, the app will still run stage-1 and show a warning for stage-2 classification.
- You can provide a JSON class mapping in the sidebar (for example numeric ID to class name).
