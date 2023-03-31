import json
from locust import events, HttpUser, task


# Retrieve command line arguments
# Locust docs: https://docs.locust.io/en/stable/extending-locust.html?highlight=event#custom-arguments
@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--endpoint-name", default="")
    parser.add_argument("--databricks-pat", is_secret=True, default="")

class LoadTestUser(HttpUser):

# Reads features from JSON file. Format expected by Databricks Model Serving
# explained here: https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#request-format
    with open("local_load_test/features.json", "r") as json_features:
        model_input = json.load(json_features)
    
    @task
    def query_single_model(self):
        
        token = self.environment.parsed_options.databricks_pat
        endpoint_name = self.environment.parsed_options.endpoint_name

        headers = {"Authorization": f"Bearer {token}"}
       
        self.client.post(f"/serving-endpoints/{endpoint_name}/invocations",
                         headers=headers,
                         json=self.model_input)
