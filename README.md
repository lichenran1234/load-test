# How to Load Test Databricks Model Serving Endpoints Using Locust

Load testing [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html) endpoints is an important step before moving the endpoint to production. A load test verfies the latency meets your requirements, helps you estimate costs, and determines expected throughput and concurrency. 

This repository demonstrates how to load test a Databricks Model Serving endpoint using the open source load testing tool [locust.io](https://locust.io/).

## RequirementsAbility to handle difficult customer situations Ability to speak to different audiences Ability to think on your feet Clear Communication Succinct Answers

Before starting the walkthrough, make sure you completed the following tasks:

- [Create a Model Serving endpoint](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#ui-workflow) and verify it is in the "Ready" state.
- [Record the instance name](https://docs.databricks.com/workspace/workspace-details.html#workspace-instance-names-urls-and-ids) of the Databricks workspace where the model is deployed.
- Databricks APIs use Personal Access Tokens (PATs) to authenticate. Use an existing PAT or [generate a new PAT](https://docs.databricks.com/dev-tools/auth.html#personal-access-tokens-for-users) that has "Can View" or "Can Manage" permissions on the model serving endpoint. 
- If your Databricks workspace restricts IP addresses to an IP access list, your load testing source's IP address must be within the access list. You may need to connect to your company's VPN. You can check if IP access list are in use in your Databricks workspace by [following the instructions here](https://docs.databricks.com/security/network/ip-access-list.html#check-if-your-workspace-has-the-ip-access-list-feature-enabled).

## Setting up Locust on your own computer in single-process mode (5 min setup, supports low QPS)

Follow [the steps outlined here](/local-load-test/README.md) to run a single-process load test from your local computer. Locust will only use one CPU core on your machine, and the max QPS supported depends on your payload size. For small payloads, it can support up to hundreds of QPS. For large payloads, it may only support less than 10 QPS.


## Setting up distributed Locust on a powerful machine (30 min setup, supports high QPS)
Follow [the steps outlined here](/distributed-load-test/README.md) to set up distributed Locust on an AWS EC2 instance. It should also be applicable to other cloud providers’ Ubuntu instances.