from locust import HttpUser, task

class LoadTestUser(HttpUser):
    # please update the following host
    host = "https://xxxxxxx.cloud.databricks.com"
    
    # please update the following header
    headers = {'Authorization': 'Bearer dapixxxxxxx'}
    
    # please update the following model input
    model_input = {"dataframe_split": { "columns": ["feature0", "feature1"], "data": [[0, 1]]} }

    @task
    def query_single_model(self):
        # please update the following endpoint name
        self.client.post("/serving-endpoints/MY_ENDPOINT_NAME/invocations",
                         headers=self.headers,
                         json=self.model_input)