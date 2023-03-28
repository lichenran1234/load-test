### Step 1: Launch an EC2 instance
* The recommended AMI is __Ubuntu Server 20.04 LTS (HVM), SSD Volume Type__, as the following instructions have been tested on it.
* For the instance type, start with __c5.4xlarge__ (with 16 vCPUs). You can choose other instance types based on your needs (for example, c5.24xlarge with more vCPUs). In general, compute-optimized instances (C-family) are recommended for load testing. More CPUs support higher QPS.
* Remember to allocate disk space for the instance (by attaching a volume). __100 GB__ of disk space should be sufficient.
* Remember to make __port 80__ of the instance reachable from your laptop (by configuring the __security group__) and enabling __Auto-assign public IP__. These steps are necessary to access the Locust UI later on.

### Step 2: Install dependencies needed to run Locust

ssh into the EC2 instance. Install Docker and Docker Compose by running the following commands one by one:

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

### Step 3: Prepare load test files

Clone this GitHub repository to copy its content to the instance. Navigate to the cloned repository's folder. Overwrite the high_qps_load_test/features.json file with the sample payload for you model.

### Step 4: Start Locust and run load tests

In the cloned folder, run `sudo docker-compose up --scale worker=16` to start Locust. Note that `worker=16` means there will be 16 Locust workers (utilizing 16 CPU cores on the machine). It’s recommended to have `#workers == #vCPUs-on-the-machine`.

After Locust starts, visit `http://{your-ec2-instance-ip}` (use http, __not__ https) from your laptop to access the Locust UI, where you can easily start/stop a load test. Remember that we opened port 80 of the EC2 instance to our laptop, which is why we can visit the Locust UI.

### Step 5: Cleanup

Don’t forget to terminate the EC2 instance after you finish the load tests to minimize cloud costs.