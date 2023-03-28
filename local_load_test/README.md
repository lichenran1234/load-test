### Step 1: Install Locust on your own computer

Follow the [official locust docs](https://docs.locust.io/en/stable/installation.html) to install locust on your local computer.

### Step 2: Copy the repository's content

Clone this repository's content to your local computer. Open a terminal window and navigate to the cloned directory.

### Step 3: Run the load test

Run the command `locust -f local_load_test/load_test.py` to start the locust program and UI. Visit http://0.0.0.0:8089 on your computer. The UI looks like this:
![image1](https://user-images.githubusercontent.com/5786126/222304686-0f9fd25e-d168-4625-8ced-b45b22e697fa.png)

Enter the peak test concurrency for "Number of users". Enter the Databricks instance name for "Host" (ex: https://\<workspace-specific-identifier\>.cloud.databricks.com). Enter the model serving endpoint name for "Endpoint name". Copy your Databricks PAT in to the "Databricks pat" field. Press "Start swarming" to start the load test!