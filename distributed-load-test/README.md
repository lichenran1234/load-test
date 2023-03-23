### Step 1: start an EC2 instance
* The recommended AMI is __Ubuntu Server 20.04 LTS (HVM), SSD Volume Type__, as the following instructions have been tested on it.
* For the instance type, start with __c5.4xlarge__ (with 16 vCPUs). You can choose other instance types based on your needs (for example, c5.24xlarge with more vCPUs). In general, compute-optimized instances (C-family) are recommended for load testing. More CPUs support higher QPS.
* Remember to allocate some disk space for the instance (by attaching a volume). __100 GB__ of disk space should be sufficient.
* Remember to make the __80 port__ of the instance reachable from your laptop (by configuring the __security group__). It’s needed to access the load test UI later on.

### Step 2: install dependencies needed to run locust
#### Step 2.1: ssh into the EC2 instance
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

### Step 3: write a locust file `load_test.py` with the load test logics

In the `load-test` folder, create a file named `load_test.py` with the following content to run load tests against Databricks Model Serving (alternatively, read the [official locust doc](https://docs.locust.io/en/stable/writing-a-locustfile.html) to learn how to write a locust file):

### Step 4: start locust and run load tests
In the `load-test` folder, run `sudo docker-compose up --scale worker=16` to start locust. Note that `worker=16` means there will be 16 locust workers (utilizing 16 CPU cores on the machine). It’s recommended to have `#workers == #vCPUs-on-the-machine`.

After locust is started, visit `http://{your-ec2-instance-ip}` (use http, __not__ https) from your laptop to access the locust UI, where you can easily start/stop a load test. Remember that we opened the 80 port of the EC-2 instance to our laptop, which is why we can visit the locust UI from our laptop.

### Step 5: cleanup

Don’t forget to terminate the EC-2 instance after you finish the load tests.