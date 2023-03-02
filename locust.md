# Quickly setting up a load test client using Locust

This doc serves as a note on how to quickly setup a load test client that one can use to run load tests against any service (including Databricks Serverless Real-Time Inference). This is not an official Databricks doc. It’s not guaranteed that everything described here will work. We are writing this doc on a best-effort basis.

## Setting up Locust on your own computer in single-process mode (5 min setup, supports low QPS)

This is for those who just want to quickly try out a load test and don’t need very high QPS. In this mode, Locust will only be able to utilize one CPU core on your machine, and the max QPS supported by one single CPU depends on your payload size. For small payloads, it can support up to hundreds of QPS. For large payloads, it may only support less than 10 QPS.

### Step 1: install Locust on your own computer

Follow the [official doc]([url](https://docs.locust.io/en/stable/installation.html)) for installation.

### Step 2: write a locust file `load_test.py` with the load test logics

Create a file named `load_test.py` with the following content:

```python
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
                         headers=headers,
                         json=model_input)
```

### Step 3: run the load test

Inside the folder where you have the `load_test.py` file, run the command `locust -f load_test.py`. That will start the locust program and UI. Visit http://0.0.0.0:8089 on your computer and you’ll be able to start the load test. The UI looks like this:
![image1](https://user-images.githubusercontent.com/5786126/222304686-0f9fd25e-d168-4625-8ced-b45b22e697fa.png)


## Setting up distributed Locust on a powerful machine (30 min setup, supports high QPS)
The following instructions tell you how to set up distributed Locust on an AWS EC-2 instance (with Ubuntu 20.04). It should also be applicable to other cloud providers’ Ubuntu instances.

Assuming you already know how to create EC-2 instances on AWS, here are the detailed instructions.

### Step 1: start an EC-2 instance
* The recommended AMI is __Ubuntu Server 20.04 LTS (HVM), SSD Volume Type__, as the following instructions have been tested on it.
* For the instance type, it’s not a bad idea to start with c5.4xlarge (with 16 vCPUs). You can choose other instance types based on your needs (for example, c5.24xlarge with more vCPUs). In general, compute-optimized instances (C-family) are recommended for load testing. More CPUs support more QPS.
* Remember to allocate some disk space for the instance (by attaching a volume). 100 GB of disk space should be sufficient.
* Remember to make the __80 port__ of the instance reachable from your laptop (by configuring the __security group__). It’s needed to access the load test UI later on.

### Step 2: install dependencies needed to run locust
#### Step 2.1: ssh into the EC-2 instance
#### Step 2.2: create a folder named `load-test` and cd into it
#### Step 2.3: install docker and docker-compose by running the following commands one by one:

```bash
sudo apt update

sudo apt -y install apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"

sudo apt update

sudo apt -y install docker-ce

sudo curl -L "https://github.com/docker/compose/releases/download/v2.1.1/docker-compose-linux-x86_64" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose
```

#### Step 2.4: in the same `load-test` folder, create a file named `docker-compose.yml` with the following content:

```yaml
version: '3'

services:
  master:
    image: locustio/locust
    ports:
      - "80:8089"
    volumes:
      - ./:/mnt/locust
    command: -f /mnt/locust/load_test.py --master

  worker:
    image: locustio/locust
    volumes:
      - ./:/mnt/locust
    command: -f /mnt/locust/load_test.py --worker --master-host master
```

### Step 3: write a locust file “load_test.py” with the load test logics

In the `load-test` folder, create a file named `load_test.py` with the following content to run load tests against Databricks Model Serving (alternatively, read the [official locust doc]([url](https://docs.locust.io/en/stable/writing-a-locustfile.html)) to learn how to write a locust file):

```python
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
                         headers=headers,
                         json=model_input)
```

### Step 4: start locust and run load tests
In the `load-test` folder, run `sudo docker-compose up --scale worker=16` to start locust. Note that `worker=16` means there will be 16 locust workers (utilizing 16 CPU cores on the machine). It’s recommended to have `#workers == #vCPUs-on-the-machine`.

After locust is started, visit `http://{your-ec2-instance-ip}` (__http, not https__) from your laptop to access the locust UI, where you can easily start/stop a load test. Remember that we opened the 80 port of the EC-2 instance to our laptop, which is why we can visit the locust UI from our laptop.

### Step 5: cleanup

Don’t forget to terminate the EC-2 instance after you finish the load tests.



