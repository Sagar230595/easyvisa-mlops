# EasyVisa Streamlit app

A single-page UI that loads the champion model from the Unity Catalog registry
and predicts visa certification for one application.

## Run locally
```
pip install -r app/requirements.txt
# authenticate to Databricks (for the UC model registry):
export DATABRICKS_HOST=https://<workspace>.databricks.net
export DATABRICKS_TOKEN=<token>
streamlit run app/streamlit_app.py
```

## Deploy as a Databricks App
```
databricks apps create easyvisa-predictor
databricks sync . /Workspace/Users/<you>/easyvisa-mlops   # or use the bundle
databricks apps deploy easyvisa-predictor \
  --source-code-path /Workspace/Users/<you>/easyvisa-mlops
```
The app's service principal needs EXECUTE on the model
`<catalog>.<schema>.easyvisa_visa_approval` and USE on the catalog/schema.
`app.yaml` defines the start command.
