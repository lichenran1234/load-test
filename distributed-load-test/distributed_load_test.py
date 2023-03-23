from locust import events, HttpUser, task

@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--endpoint-name", default="")
    parser.add_argument("--databricks-pat", is_secret=True, default="")

class LoadTestUser(HttpUser):
    
    model_input = {"dataframe_split": {
              "index": [0],
              "columns": ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulphur_dioxide", "density", "pH", "sulphates", "alcohol"],
             "data": [
              [2.0, 6, 3, 2.4, 12, 4, 12, 8, 3, 2.5, 1.0]
             ]
            }
          }

    @task
    def query_single_model(self):
        token = self.environment.parsed_options.databricks_pat
        endpoint_name = self.environment.parsed_options.endpoint_name

        headers = {"Authorization": f"Bearer {token}"}
        
        self.client.post(f"/serving-endpoints/{endpoint_name}/invocations",
                         headers=headers,
                         json=self.model_input)