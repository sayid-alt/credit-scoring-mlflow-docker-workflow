import mlflow
from mlflow.models import validate_serving_input
from mlflow.models import convert_input_example_to_serving_input
import pandas as pd
import json

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
model_uri = "runs:/00138917787d4c5aa6e016fd33b0770b/model"


# Validate the serving input example
with open("serving_input_example.json", "r") as f:
    data = json.load(f)
    
result_valid = validate_serving_input(model_uri, data)
print(result_valid)


# Inference
model = mlflow.pyfunc.load_model(model_uri)
data = pd.DataFrame(data['dataframe_split']['data'], columns=data['dataframe_split']['columns'])
predictions = model.predict(data)
print(predictions)