# How to Load Test Databricks Model Serving Endpoints

## Load Testing using Databricks Notebook (no setup needed)
This [notebook](/db_load_testing_notebook) provides a convenient way to load test your serving endpoints and obtain insights into workload size, QPS, and latency. By using the out-of-the-box setup, you can quickly get started with load testing without any additional setup. This notebook is recommended for low to mid QPS goals. If you have a QPS goal greater than 2k, we recommend following the rest of this tutorial to set up Locust.

## Load Testing using Locust
Load testing [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html) endpoints is an important step before moving the endpoint to production. A load test verfies the latency meets your requirements, helps you estimate costs, and determines expected throughput and concurrency. 

This repository demonstrates how to load test a Databricks Model Serving endpoint using the open source load testing tool [locust.io](https://locust.io/).

## Expected Results

After setting up Locust, you will be able to send requests to your model endpoint with a configurable concurrency. Locust will record the response latency (p50, p75, p99, ect) and display it via the web UI. The data can also be downloaded in CSV format.

![Screen Shot 2023-03-29 at 10 39 07 AM](https://user-images.githubusercontent.com/93339895/228624340-4cec3835-4d2d-4ff9-8fa6-24ec2b6197c4.png)

## Requirements

Before starting the walkthrough, make sure you complete the following tasks:

- [Create a Model Serving endpoint](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#ui-workflow) and verify it is in the "Ready" state.
- Have at least one sample payload for the model ready in JSON format. You can see the [supported format options here](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#request-format).
- [Record the instance name](https://docs.databricks.com/workspace/workspace-details.html#workspace-instance-names-urls-and-ids) of the Databricks workspace where the model is deployed.
- Databricks APIs use Personal Access Tokens (PAT)s to authenticate. Use an existing PAT or [generate a new PAT](https://docs.databricks.com/dev-tools/auth.html#personal-access-tokens-for-users) that has "Can View" or "Can Manage" permissions on the model serving endpoint. 
- If your Databricks workspace restricts IP addresses to an IP access list, your load testing source's IP address must be within the access list. You may need to connect to your company's VPN. You can check if IP access lists are used in your Databricks workspace by [following the instructions here](https://docs.databricks.com/security/network/ip-access-list.html#check-if-your-workspace-has-the-ip-access-list-feature-enabled).

## Setting up Locust on your own computer in single-process mode (5 min setup, supports low QPS)

Follow [the steps outlined here](/local_load_test/README.md) to run a single-process load test from your local computer. Locust will only use one CPU core on your machine, and the max queries per second (QPS) supported depends on your payload size. For small payloads, it can support up to hundreds of QPS. For large payloads, it may only support less than 10 QPS.

## Setting up distributed Locust on a powerful machine (30 min setup, supports high QPS)
Follow [the steps outlined here](/high_qps_load_test/README.md) to set up distributed Locust on an AWS EC2 instance. It should also be applicable to other cloud providers’ Ubuntu instances.
