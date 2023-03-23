import json
import pandas as pd
from locust import events, HttpUser, task


# Retrieve command line arguments
# Locust docs: https://docs.locust.io/en/stable/extending-locust.html?highlight=event#custom-arguments
@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--endpoint-name", default="")
    parser.add_argument("--databricks-pat", is_secret=True, default="")

# Reads features from CSV file and formats according to Databricks model
# serving spec: https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#request-format
def read_features(path="local-load-test/features.csv"):

    features = pd.read_csv(path)
    features_json = json.loads(features.to_json(path_or_buf=None, 
                                                orient="split"))
  
    return {"dataframe_split": features_json}

class LoadTestUser(HttpUser):

    model_input = read_features()
   
    @task
    def query_single_model(self):

        token = self.environment.parsed_options.databricks_pat
        endpoint_name = self.environment.parsed_options.endpoint_name

        headers = {"Authorization": f"Bearer {token}"}
       
        self.client.post(f"/serving-endpoints/{endpoint_name}/invocations",
                         headers=headers,
                         json=self.model_input)
