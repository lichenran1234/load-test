# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Databricks Model Serving Load Testing Notebook
# MAGIC 
# MAGIC This notebook helps you find the best:
# MAGIC 1) workload size (e.g. "Small", "Medium", "Large") for your Databricks Model Serving endpoint, given your QPS goal.
# MAGIC 2) QPS/latency tradeoff
# MAGIC 
# MAGIC If your QPS goal is greater than 2k, please consider using a professional load testing tool instead. For example, see [this tutorial](https://github.com/lichenran1234/load-test/blob/main/README.md) for setting up Locust.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Instructions
# MAGIC 
# MAGIC 1. Attach this notebook to a __single-node__ Spark cluster (__compute-optimized__ instances are __highly__ recommended) with __X__ cores (general guideline is __X__ cores can support __X * 100__ QPS for load tests). This notebook can utilize multiple CPU cores on a single node, but not multiple nodes in the same cluster.
# MAGIC 2. To better simulate the reality, launch this Spark cluster in the __same__ region as your application that calls the Model Serving endpoint. Because the physical distance between client and server can significantly affect the latency.
# MAGIC 3. When executing this notebook, read the output of each cell carefully before moving on to the next one.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Quick Diagnosis of Failed Requests
# MAGIC 
# MAGIC | HTTP Response Code | Error Message in Response Body | What to do |
# MAGIC | ----------- | ----------- | ------- |
# MAGIC | 400      | Failed to parse input...       | Validate your input |
# MAGIC | 403   | Invalid access token        | Check if your token is valid |
# MAGIC | 404 | RESOURCE_DOES_NOT_EXIST | Check that your endpoint exists and is in "Ready" state |
# MAGIC | 429 | Exceeded max number of parallel requests | Reach out to Databricks to get more __parallel request quota__ |
# MAGIC | 429 | Too Many Requests | Reach out to Databricks to get more __QPS quota__ |
# MAGIC | 504 | timeout | Check if your model takes more than 60s to execute, if so, reach out to Databricks to raise the model execution timeout limit |
# MAGIC | 500 or 503 | ... | Reach out to Databricks. |

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Step 1: set parameters

# COMMAND ----------

# DBTITLE 1,Run this cell to import libraries
import requests
from threading import Thread
from threading import Lock
import time
from collections import defaultdict
import numpy as np
import json
import psutil
import multiprocessing
from multiprocessing import Process
import datetime
import os
import pickle
import pandas as pd
import math

# COMMAND ----------

# DBTITLE 1,Set the parameters, and run this cell
# String. The full URL of the endpoint to load test against. For example: ENDPOINT = "https://xxx.cloud.databricks.com/serving-endpoints/xxx/invocations"
ENDPOINT = ""

# String. The token to use for this load test. For example: TOKEN = "dapide256eb400d7130b4c874bd63e6e89d2"
TOKEN = ""

# String. The JSON string to use as the model input. For example: DATA = json.dumps({"dataframe_split": {"columns": ["col1", "col2"], "data": [[1, 2]]}})
DATA = ""

# Integer. The target of your QPS. For example: QPS_TARGET = 2000
QPS_TARGET = 100

# COMMAND ----------

# DBTITLE 1,Run this cell to validate the parameters above
session = requests.Session()
req = requests.Request('POST', ENDPOINT, headers={'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}, data=DATA)
prepped = req.prepare()

num_req = 5

print(f"Sending {num_req} requests to the endpoint as a sanity check.\n")

has_failed_request = False
for i in range(num_req):
  resp = session.send(prepped)
  if resp.status_code != 200:
    has_failed_request = True
    print("Request failed. Please DO NOT move to the next step. Check if the token/endpoint/data are all valid.")
    print("  response code: " + str(resp.status_code))
    print("  response body: " + str(resp.text))
    break
  if i == 0:
    print("First request succeeded, please sanity-check the response: " + str(resp.text))
if not has_failed_request:
  print(f"\nCongratulations! All {num_req} requests succeeded. Please proceed to the next step.")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Step 2: decide the duration of the load test

# COMMAND ----------

# DBTITLE 1,Run this cell to estimate the duration of the load test
# Decide the duration of the load test
num_req = 100
print("Sending %d requests to warm up the endpoint..." % num_req)
for i in range(num_req):
  session.send(prepped)

print("Sending %d requests to measure the latency..." % num_req)
latencies_seconds = []
for i in range(num_req):
  start = time.perf_counter()
  resp = session.send(prepped)
  end = time.perf_counter()
  latencies_seconds.append((end - start))

latency_p50 = np.percentile(latencies_seconds, 50)

NUM_REQUESTS_EACH_THREAD = 2000

if (latency_p50 > 4):
  estimated_qps_for_64_provisioned_concurrency = 64 / latency_p50
  print("\nThe P50 latency is roughly %.2f seconds, which is too high for real-time inference. \
Even the largest workload size (with 64 provisioned concurrency) can only support roughly %.2f \
QPS. Are you sure you want to run a load test on this model? Feel free to reach out to Databricks \
for possible optimizations on the model to reduce latency." % (latency_p50, estimated_qps_for_64_provisioned_concurrency))
else:
  latency_p50_ms = latency_p50 * 1000
  estimated_duration_in_seconds_for_each_run = NUM_REQUESTS_EACH_THREAD * latency_p50
  print("\nThe P50 latency is roughly %d ms. In each load test run we'll launch a few threads, \
and each thread will send %d requests sequentially. That means each load test run will take around \
%d * %d / 1000 = %d seconds. And we need ~10 runs to finish the whole load test. \
Is that okay?" % (latency_p50_ms, NUM_REQUESTS_EACH_THREAD, NUM_REQUESTS_EACH_THREAD, latency_p50_ms, estimated_duration_in_seconds_for_each_run))
  
  print("  If it's okay, do not change the 'NUM_REQUESTS_EACH_THREAD' param in the next cell.")
  print("  Otherwise reduce 'NUM_REQUESTS_EACH_THREAD' to make the tests faster, but the test result will be less accurate.")

# COMMAND ----------

# DBTITLE 1,Set the parameter, and run this cell
NUM_REQUESTS_EACH_THREAD = 2000

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Step 3: load test w/ "Small" workload size w/o scale-to-zero
# MAGIC 
# MAGIC Please use __"Small"__ workload size (__4__ provisioned concurrency) for your endpoint, and __disable__ scale-to-zero.
# MAGIC 
# MAGIC Make sure your endpoint is __Ready__ with __"Small"__ workload size before performing the load test!

# COMMAND ----------

# DBTITLE 1,Run this cell to perform the load test
def load_test_process(process_id, num_threads, output_dir):
  session = requests.Session()
  # TCP connection pool size == num_threads_for_this_process
  adapter = requests.adapters.HTTPAdapter(pool_connections=num_threads, pool_maxsize=num_threads)
  session.mount('https://', adapter)
  req = requests.Request('POST', ENDPOINT, headers={'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}, data=DATA)
  prepped = req.prepare()

  timestamp_to_latency_ms_dict = defaultdict(list)
  response_code_count_dict = defaultdict(int)
  error_code_to_error_message_dict = defaultdict(set)
  
  lock = Lock()
  
  num_running_threads = [0]
  class LoadTestThread(Thread):
    def run(self):
      # warm up
      for i in range(NUM_REQUESTS_EACH_THREAD // 10):
        session.send(prepped)
      
      with lock:
        num_running_threads[0] += 1
      for i in range(NUM_REQUESTS_EACH_THREAD):
        start = time.perf_counter()
        resp = session.send(prepped)
        end = time.perf_counter()

        response_code_count_dict[resp.status_code] += 1
        if resp.status_code == 200:
          # Only record the latency of successful requests.
          # Record latency at millisecond level, so multiply `(end - start)` by 1e3
          timestamp_to_latency_ms_dict[int(start)].append((end - start) * 1000)
        else:
          error_code_to_error_message_dict[resp.status_code].add(str(resp.text))
      with lock:
        num_running_threads[0] -= 1

  cpu_util_stats = []
  class CpuMonitoringThread(Thread):
    def run(self):
      started = False
      while True:
        if not started:
          if num_running_threads[0] == num_threads:
            started = True
            continue
        else:
          if num_running_threads[0] == 0:
            # Exit when all load testing threads have finished.
            break          
          total = psutil.cpu_percent(interval=0.5, percpu=True)
          num_cores = len(total)
          avg = sum(total) / num_cores
          cpu_util_stats.append(avg)

  threads = []
  if process_id == 0:
    threads.append(CpuMonitoringThread())
    threads[0].start()

  for i in range(num_threads):
    t = LoadTestThread()
    t.start()
    threads.append(t)

  for t in threads:
    t.join()
  
  output_dir = output_dir + "/" + str(process_id)
  os.mkdir(output_dir)
  with open(output_dir + '/result', 'wb') as f:
    pickle.dump(timestamp_to_latency_ms_dict, f)
  with open(output_dir + '/response_code', 'wb') as f:
    pickle.dump(response_code_count_dict, f)
  with open(output_dir + '/error_message', 'wb') as f:
    pickle.dump(error_code_to_error_message_dict, f)
  if process_id == 0:
    with open(output_dir + '/cpu_stats', 'wb') as f:
      pickle.dump(cpu_util_stats, f)
# ====================== END OF def load_test_process(process_id, num_threads, output_dir):


def start_a_load_test_run(output_dir, run_id, num_threads_total):
  # Start one process for each core to fully utilize all CPUs.
  num_processes = psutil.cpu_count()
  # Evenly distribute all threads onto each process.
  num_threads_for_each_process = [num_threads_total // num_processes] * num_processes
  for i in range(num_threads_total % num_processes): num_threads_for_each_process[i] += 1

  # Launch processes.
  processes = []
  for i in range(num_processes):
    num_threads = num_threads_for_each_process[i]
    if num_threads > 0:
      p = Process(target=load_test_process, args=(i, num_threads, output_dir))
      p.start()
      processes.append(p)
  for p in processes:
    p.join()
# =================== END OF def start_a_load_test_run(num_threads_total):


def parse_result(per_run_output_dir):
  files = os.listdir(per_run_output_dir)
  files = [os.path.join(per_run_output_dir, f) for f in files]
  subdirs = [f for f in files if not os.path.isfile(f)]

  timestamp_to_latency_ms_dict = defaultdict(list)
  response_code_count_dict = defaultdict(int)
  error_code_to_error_message_dict = defaultdict(set)
  for subdir in subdirs:
    with open(subdir + '/result', 'rb') as f:
      cur_dict = pickle.load(f)
      for key in cur_dict:
        timestamp_to_latency_ms_dict[key] += cur_dict[key]
    with open(subdir + '/response_code', 'rb') as f:
      cur_dict = pickle.load(f)
      for key in cur_dict:
        response_code_count_dict[key] += cur_dict[key]
    with open(subdir + '/error_message', 'rb') as f:
      cur_dict = pickle.load(f)
      for key in cur_dict:
        error_code_to_error_message_dict[key] |= cur_dict[key]

  total_requests = sum(response_code_count_dict.values())
  success_rate = response_code_count_dict[200] / total_requests
  
  err_msg = ""
  if success_rate < 1:
    err_msg += "There were failed requests. Here is the distribution of HTTP response codes:\n"
    err_msg += "  Total number of requests: %d\n" % total_requests
    for key in response_code_count_dict:
      err_msg += "    HTTP response code %d: %.2f%%\n" % (key, response_code_count_dict[key] / total_requests * 100)
    err_msg += "  Here are the HTTP response body of the failed requests:\n"
    for key in error_code_to_error_message_dict:
      err_msg += "    response code: %d\n" % (key)
      err_msg += "    response body: %s\n" % (error_code_to_error_message_dict[key])
  is_result_valid = success_rate >= 0.95
  return is_result_valid, timestamp_to_latency_ms_dict, err_msg
# =================== END OF def parse_result(per_run_output_dir):


def parse_qps_and_latency(timestamp_to_latency_ms_dict):
  assert len(timestamp_to_latency_ms_dict) >= 2, "The load test run duration should exceed 2 seconds!"
  duration_seconds = max(timestamp_to_latency_ms_dict.keys()) - min(timestamp_to_latency_ms_dict.keys())
  
  qps_list = [len(timestamp_to_latency_ms_dict[x]) for x in timestamp_to_latency_ms_dict]
  qps_threshold = np.percentile(qps_list, 50) * 0.8
  
  new_dict = {}
  total_seconds = 0
  latency_list = []
  for key, value in timestamp_to_latency_ms_dict.items():
    if len(value) > qps_threshold:
      new_dict[key] = value
      total_seconds += 1
      latency_list += new_dict[key]
  avg_qps = len(latency_list) / total_seconds
  latency_p50 = np.percentile(latency_list, 50)
  latency_p90 = np.percentile(latency_list, 90)
  latency_p99 = np.percentile(latency_list, 99)
  return new_dict, duration_seconds, avg_qps, latency_p50, latency_p90, latency_p99


def run_load_test_against_small_workload_size():
  NUM_THREADS_FOR_EACH_RUN = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36]
  timestamp = datetime.datetime.now().strftime("load-test-UTC-%Y-%m-%d-%H-%M-%S")
  output_dir = "/tmp/load_tests/" + timestamp
  os.makedirs(output_dir)
  
  run_stats = {}
  qps_saturated = False
  for i in range(len(NUM_THREADS_FOR_EACH_RUN)):
    num_threads = NUM_THREADS_FOR_EACH_RUN[i]
    per_run_output_dir = output_dir + "/run_%d_threads_%d" % (i, num_threads)
    os.mkdir(per_run_output_dir)
    print("\nStarting run_%d with %d threads..." % (i, num_threads))
    start_a_load_test_run(output_dir=per_run_output_dir, run_id=i, num_threads_total=num_threads)
    print("Finished run_%d" % i)
    
    # Check if there were too many failed requests.
    is_result_valid, timestamp_to_latency_ms_dict, err_msg = parse_result(per_run_output_dir)
    if err_msg: print(err_msg)
    assert is_result_valid, "Exiting the load test due to too many failed requests during run_%d, please debug the failed requests and rerun the load test." % i
    
    new_dict, duration_seconds, avg_qps, latency_p50, latency_p90, latency_p99 = parse_qps_and_latency(timestamp_to_latency_ms_dict)
    print("Result of run_%d: duration %d seconds, avg_qps %.2f, latency_p50 %.1f ms, latency_p90 %.1f ms, latency_p99 %.1f ms" % \
          (i, duration_seconds, avg_qps, latency_p50, latency_p90, latency_p99))
    
    # Check CPU utilization during this run.
    with open(per_run_output_dir + '/0/cpu_stats', 'rb') as f:
      cpu_stats = pickle.load(f)
      threshold = np.percentile(cpu_stats, 50) * 0.5
      valid_cpu_stats = list(filter(lambda x: x > threshold, cpu_stats))
      p80 = np.percentile(valid_cpu_stats, 80)
      avg = sum(valid_cpu_stats) / len(valid_cpu_stats)
      assert p80 < 70, "\nCPU utilization was too high during the load test!!! The avg \
CPU utilization across all %d cores was %.1f%%. The load test result will be meaningless \
under this condition. Please launch a Spark cluster with more CPUs and rerun the load test." % (psutil.cpu_count(), avg)

    run_stats[i] = new_dict, num_threads, avg_qps, latency_p50, latency_p90, latency_p99, duration_seconds
    if avg_qps > QPS_TARGET:
      print("\nThe QPS of this run (%.2f) exceeded your QPS target (%d) already! Exiting the load test..." % (avg_qps, QPS_TARGET))
      break
    
    if qps_saturated:
      break
    if i >= 3:
      if avg_qps < run_stats[i-1][2]: qps_saturated = True
      elif (avg_qps - run_stats[i-1][2]) / run_stats[i-1][2] < 0.05: qps_saturated = True
  print("Finished the load test successfully!")
  return output_dir, run_stats
# =================== END OF def run_load_test_against_small_workload_size():


output_dir, run_stats = run_load_test_against_small_workload_size()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3.1 - check the load test result
# MAGIC 
# MAGIC We fixed the __provisioned concurrency at 4__ throughout the load test. In each run, we adjusted __#client-threads__ (i.e. __#parallel-requests__) to see how it affects the QPS and latency. Now go to the next cell to see the summary of the test results.
# MAGIC 
# MAGIC #### Note: client-side stats vs. server-side stats
# MAGIC This notebook reports __client-side__ stats, while Databricks Model Serving UI reports __server-side__ stats.
# MAGIC * Client and server report the __same QPS stats__ but __different latency stats__
# MAGIC * The difference between __client-side__ latency and __server-side__ latency is decided by the __physical distance__ between client and srever
# MAGIC * Databricks UI reports QPS/latency stats at __per-minute__ level, while this notebook records QPS/latency at __per-second__ level.

# COMMAND ----------

# DBTITLE 1,Run this cell to print a summary of the load test runs
cols = ["run_id", "#parallel_requests", "duration (seconds)", "avg qps for the whole run", "latency_ms_p50", "latency_ms_p90", "latency_ms_p99"]
data = []
num_parallel_requests_list = []
qps_list = []
latency_p50_list = []
latency_p90_list = []
latency_p99_list = []
for i in sorted(run_stats.keys()):
  timestamp_to_latency_ms_dict, num_threads, avg_qps, latency_p50, latency_p90, latency_p99, duration_seconds = run_stats[i]
  data.append([i, num_threads, duration_seconds, "%d" % avg_qps, "%d" % latency_p50, "%d" % latency_p90, "%d" % latency_p99])
  num_parallel_requests_list.append(num_threads)
  qps_list.append(avg_qps)
  latency_p50_list.append(latency_p50)
  latency_p90_list.append(latency_p90)
  latency_p99_list.append(latency_p99)
df = pd.DataFrame(data=data, columns=cols)
display(df)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.xlabel("#parallel-requests")
plt.ylabel("QPS")
l1, = ax.plot(num_parallel_requests_list, qps_list, label="QPS")
ax2 = ax.twinx()
l2, = ax2.plot(num_parallel_requests_list, latency_p50_list, 'C1', label="latency P50 (ms)")
l2, = ax2.plot(num_parallel_requests_list, latency_p90_list, 'C2', label="latency P90 (ms)")
l2, = ax2.plot(num_parallel_requests_list, latency_p99_list, 'C3', label="latency P99 (ms)")

ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', borderaxespad=0.)
ax2.legend(bbox_to_anchor=(1., 1.02, 1., .102), loc='lower left', borderaxespad=0.)

plt.title('QPS/Latency vs. #parallel-requests')
plt.ylabel("Latency")
plt.grid(True)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### How to interpret the graph above
# MAGIC Overall, when fixing the __provisioned concurrency at 4__ throughout the load test, as we increase #parallel-requests, __both__ the QPS and latency will increase.
# MAGIC * when increasing #parallel-requests between __1 - 4__:
# MAGIC   * QPS increases __fast__
# MAGIC   * latency increases __slowly__
# MAGIC   * because queuing effect is not significant at this stage (4 provisioned concurrency means 4 workers, which can handle at most 4 parallel requests at any time)
# MAGIC * when increasing #parallel-requests beyond __4__:
# MAGIC   * QPS starts to increase __more slowly__
# MAGIC   * latency starts to increase __faster__
# MAGIC   * queuing effect starts to get more significant
# MAGIC * when increasing #parallel-requests beyond __X__ (X is the largest #parallel-requests in the graph above):
# MAGIC   * QPS almost __stops__ to increase
# MAGIC   * latency starts to increase __drastically__
# MAGIC   * queuing effect gets even more significant
# MAGIC 
# MAGIC #### How to choose the best parallel_request/provisioned_concurrency ratio
# MAGIC * If you want the best latency, choose N:4 (N <= 4)
# MAGIC * If you are willing to trade some latency for better QPS, choose N:4 (4 < N < X)
# MAGIC * Never choose N:4 (N >= X)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3.2 - choose the best run_id
# MAGIC 
# MAGIC The load test above might not hit your QPS target, but that's okay because we haven't tried "Medium" or "Large" workload sizes yet. Now we need to know your preferred QPS/latency tradeoff to decide the best __parallel_request : provisioned_concurrency__ ratio.
# MAGIC 
# MAGIC Please choose the best run_id above and set `BEST_RUN_ID` in the next cell.
# MAGIC * The best run_id should represent your preferred QPS/latency tradeoff: choose one with relatively high QPS and relatively low latency.

# COMMAND ----------

# DBTITLE 1,Run this cell to set the best run_id
# Set the parameter below. For example: BEST_RUN_ID = 3
BEST_RUN_ID = 3

print("You've chosen run_%d. So we'll use %d client threads for every 4 provisioned concurrency, to keep the parallel_request : provisioned_concurrency ratio at %d : 4 when trying out 'Medium' or 'Large' workload sizes." % (BEST_RUN_ID, run_stats[BEST_RUN_ID][1], run_stats[BEST_RUN_ID][1]))

# COMMAND ----------

# DBTITLE 1,Run this cell to render the per-second QPS & latency graph of the run_id you chose
run_result = run_stats[BEST_RUN_ID]
timestamp_to_latency_ms_dict = run_result[0]

timestamp_list = sorted(timestamp_to_latency_ms_dict.keys())
qps_list = [len(timestamp_to_latency_ms_dict[x]) for x in timestamp_list]
latency_p50_list = [float(np.percentile(timestamp_to_latency_ms_dict[x], 50)) for x in timestamp_list]
latency_p90_list = [float(np.percentile(timestamp_to_latency_ms_dict[x], 90)) for x in timestamp_list]

start_timestamp = min(timestamp_list)
timestamp_list = [x - start_timestamp for x in timestamp_list]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.xlabel("seconds since start of the run")
plt.ylabel("QPS")
l1, = ax.plot(timestamp_list, qps_list, label="QPS")
ax2 = ax.twinx()
l2, = ax2.plot(timestamp_list, latency_p50_list, 'C1', label="latency P50 (ms)")
l2, = ax2.plot(timestamp_list, latency_p90_list, 'C2', label="latency P90 (ms)")

ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', borderaxespad=0.)
ax2.legend(bbox_to_anchor=(1., 1.02, 1., .102), loc='lower left', borderaxespad=0.)

plt.ylabel("Latency")
plt.title('Per-second QPS & Latency')
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Step 3.3 - recommend the workload size for your QPS target

# COMMAND ----------

# DBTITLE 1,Run this cell to get the recommended workload size
qps_for_4_provisioned_concurrency = run_stats[BEST_RUN_ID][2]
parallel_requests_for_4_provisioned_concurrency = run_stats[BEST_RUN_ID][1]

if qps_for_4_provisioned_concurrency >= QPS_TARGET:
  print("Looks like the QPS for run_%d (%d) already exceeded your QPS target (%d), so the recommended workload size is 'Small' (4 provisioned concurrency)." % (BEST_RUN_ID, qps_for_4_provisioned_concurrency, QPS_TARGET))
else:
  scaling_factor = math.ceil(QPS_TARGET / qps_for_4_provisioned_concurrency)
  num_provisioned_concurrency_needed = scaling_factor * 4
  num_parallel_requsts_needed = scaling_factor * parallel_requests_for_4_provisioned_concurrency
  
  recommended_workload = ''
  workload_range = ''
  if 8 <= num_provisioned_concurrency_needed <= 16:
    recommended_workload = 'Medium'
    workload_range = '8 - 16'
  elif 16 < num_provisioned_concurrency_needed <= 64:
    recommended_workload = 'Large'
    workload_range = '16 - 64'
  
  if recommended_workload:
    print("The recommended provisioned concurrency is %d, which falls into \
the range of %s. So the recommended workload size is '%s', and the recommended \
#parallel-requests is %d. Please go to the next step to try %d parallel requests \
on '%s' workload to confirm it can hit the expected QPS of %d." % \
          (num_provisioned_concurrency_needed, workload_range, recommended_workload, num_parallel_requsts_needed, num_parallel_requsts_needed, recommended_workload, QPS_TARGET))
  else:
    print("The recommended provisioned concurrency is %d, which exceeds the upperbound (64) of the 'Large' workload. Please reach out to Databricks to get a custom workload (with %d provisioned concurrency) for the Model Serving endpoints in your workspace. If you think this is too high, please reach out to Databricks to see how the model can be optimized to reduce the latency. Reduced latency means less provisioned concurrency needed to reach the same QPS." % (num_provisioned_concurrency_needed, num_provisioned_concurrency_needed))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Step 4 - run load tests on the recommended workload size to hit your QPS target
# MAGIC 
# MAGIC Please __change__ the workload size of your endpoint as __recommended__ above, and __disable__ scale-to-zero. Make sure your endpoint is __Ready__ with the __new__ workload size before conducting the load test.

# COMMAND ----------

# DBTITLE 1,Run this cell to perform the load test (will gradually ramp up #parallel-requests)
def load_test_process_with_fixed_duration(process_id, num_threads, output_dir, duration_seconds):
  session = requests.Session()
  # TCP connection pool size == num_threads_for_this_process
  adapter = requests.adapters.HTTPAdapter(pool_connections=num_threads, pool_maxsize=num_threads)
  session.mount('https://', adapter)
  req = requests.Request('POST', ENDPOINT, headers={'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}, data=DATA)
  prepped = req.prepare()

  timestamp_to_latency_ms_dict = defaultdict(list)
  response_code_count_dict = defaultdict(int)
  error_code_to_error_message_dict = defaultdict(set)
  
  lock = Lock()
  
  num_running_threads = [0]
  class LoadTestThread(Thread):
    def run(self):
      start_time = time.perf_counter()
      
      with lock:
        num_running_threads[0] += 1
      while True:
        start = time.perf_counter()
        resp = session.send(prepped)
        end = time.perf_counter()

        response_code_count_dict[resp.status_code] += 1
        if resp.status_code == 200:
          # Only record the latency of successful requests.
          # Record latency at millisecond level, so multiply `(end - start)` by 1e3
          timestamp_to_latency_ms_dict[int(start)].append((end - start) * 1000)
        else:
          error_code_to_error_message_dict[resp.status_code].add(str(resp.text))
        if (end - start_time) > duration_seconds:
          break
      with lock:
        num_running_threads[0] -= 1

  cpu_util_stats = []
  class CpuMonitoringThread(Thread):
    def run(self):
      started = False
      while True:
        if not started:
          if num_running_threads[0] == num_threads:
            started = True
            continue
        else:
          if num_running_threads[0] == 0:
            # Exit when all load testing threads have finished.
            break          
          total = psutil.cpu_percent(interval=0.5, percpu=True)
          num_cores = len(total)
          avg = sum(total) / num_cores
          cpu_util_stats.append(avg)

  threads = []
  if process_id == 0:
    threads.append(CpuMonitoringThread())
    threads[0].start()

  for i in range(num_threads):
    t = LoadTestThread()
    t.start()
    threads.append(t)

  for t in threads:
    t.join()
  
  output_dir = output_dir + "/" + str(process_id)
  os.mkdir(output_dir)
  with open(output_dir + '/result', 'wb') as f:
    pickle.dump(timestamp_to_latency_ms_dict, f)
  with open(output_dir + '/response_code', 'wb') as f:
    pickle.dump(response_code_count_dict, f)
  with open(output_dir + '/error_message', 'wb') as f:
    pickle.dump(error_code_to_error_message_dict, f)
  if process_id == 0:
    with open(output_dir + '/cpu_stats', 'wb') as f:
      pickle.dump(cpu_util_stats, f)
# ====================== END OF def load_test_process_with_fixed_duration(process_id, num_threads, output_dir, duration_seconds):


def start_a_load_test_iteration(output_dir, num_threads_total, duration_seconds):
  # Start one process for each core to fully utilize all CPUs.
  num_processes = psutil.cpu_count()
  # Evenly distribute all threads onto each process.
  num_threads_for_each_process = [num_threads_total // num_processes] * num_processes
  for i in range(num_threads_total % num_processes): num_threads_for_each_process[i] += 1

  # Launch processes.
  processes = []
  for i in range(num_processes):
    num_threads = num_threads_for_each_process[i]
    if num_threads > 0:
      p = Process(target=load_test_process_with_fixed_duration, args=(i, num_threads, output_dir, duration_seconds))
      p.start()
      processes.append(p)
  for p in processes:
    p.join()
# =================== END OF def start_a_load_test_run(num_threads_total):


def run_load_test_against_recommended_workload_size():
  NUM_THREADS = num_parallel_requsts_needed
  NUM_ITERATIONS = 20
  DURATION_SECONDS_PER_ITER = 60

  timestamp = datetime.datetime.now().strftime("load-test-UTC-%Y-%m-%d-%H-%M-%S")
  output_dir = "/tmp/load_tests_recommended_workload/" + timestamp
  os.makedirs(output_dir)

  overall_stats_dict = defaultdict(list)
  qps_target_hit = False
  num_extra_iters_after_qps_target_hit = 2
  cur_num_threads = min(NUM_THREADS, 10)
  for i in range(NUM_ITERATIONS):
    cur_num_threads = min(cur_num_threads * 2, NUM_THREADS)
    
    per_run_output_dir = output_dir + "/run_%d" % (i)
    os.mkdir(per_run_output_dir)
    print("\nStarting iteration_%d with %d threads, estimated duration is %d seconds" % (i, cur_num_threads, DURATION_SECONDS_PER_ITER))
    start_a_load_test_iteration(per_run_output_dir, cur_num_threads, DURATION_SECONDS_PER_ITER)
    print("Finished iteration_%d" % i)

    # Check if there were too many failed requests.
    is_result_valid, timestamp_to_latency_ms_dict, err_msg = parse_result(per_run_output_dir)
    if err_msg: print(err_msg)
    assert is_result_valid, "Exiting the load test due to too many failed requests during iteration_%d, please debug the failed requests and rerun the load test." % i

    _, _, avg_qps, latency_p50, latency_p90, latency_p99 = parse_qps_and_latency(timestamp_to_latency_ms_dict)
    print("Result of iteration_%d: avg_qps %.2f, latency_p50 %.1f ms, latency_p90 %.1f ms, latency_p99 %.1f ms" % \
          (i, avg_qps, latency_p50, latency_p90, latency_p99))

    # Check CPU utilization during this run.
    with open(per_run_output_dir + '/0/cpu_stats', 'rb') as f:
      cpu_stats = pickle.load(f)
      threshold = np.percentile(cpu_stats, 50) * 0.5
      valid_cpu_stats = list(filter(lambda x: x > threshold, cpu_stats))
      p80 = np.percentile(valid_cpu_stats, 80)
      avg = sum(valid_cpu_stats) / len(valid_cpu_stats)
      assert p80 < 70, "\nCPU utilization was too high during the load test!!! The avg \
CPU utilization across all %d cores was %.1f%%. The load test result will be meaningless \
under this condition. Please launch a Spark cluster with more CPUs and rerun the load test." % (psutil.cpu_count(), avg)

    for key, value in timestamp_to_latency_ms_dict.items():
      overall_stats_dict[key] += value

    if num_extra_iters_after_qps_target_hit == 0:
      break
    if avg_qps > QPS_TARGET * 0.9:
      qps_target_hit = True
      print("\nThe QPS of this run (%.2f) was close to your QPS target (%d). Will do %d more iterations and then exit the load test..." % \
            (avg_qps, QPS_TARGET, num_extra_iters_after_qps_target_hit))
      num_extra_iters_after_qps_target_hit -= 1
  if qps_target_hit: print("Congratulations! Looks like your QPS target was hit during the load test.")
  else: print("Looks like your QPS target wasn't hit during the load test, try to rerun this load test. Reach out to Databricks for help if rerun doesn't solve the problem.")
  return overall_stats_dict

timestamp_to_latency_ms_dict = run_load_test_against_recommended_workload_size()

# COMMAND ----------

# DBTITLE 1,Run this cell to draw a graph of per-10-second QPS/latency of the load test above
# Aggregate the stats at per-10s level
aggregated_dict = defaultdict(list)
for key, val in timestamp_to_latency_ms_dict.items():
  aggregated_dict[key // 10 * 10] += val

# remove the stats for the first/last 20 seconds to de-noise the data
min_timestamp = min(sorted(aggregated_dict.keys()))
del aggregated_dict[min_timestamp]
if (min_timestamp + 10) in aggregated_dict:
  del aggregated_dict[min_timestamp + 10]
max_timestamp = max(sorted(aggregated_dict.keys()))
del aggregated_dict[max_timestamp]
if (max_timestamp - 10) in aggregated_dict:
  del aggregated_dict[max_timestamp - 10]

# prepare the data
timestamp_list = sorted(aggregated_dict.keys())
qps_list = [len(aggregated_dict[x]) // 10 for x in timestamp_list]
latency_p50_list = [float(np.percentile(aggregated_dict[x], 50)) for x in timestamp_list]
latency_p90_list = [float(np.percentile(aggregated_dict[x], 90)) for x in timestamp_list]
latency_p99_list = [float(np.percentile(aggregated_dict[x], 99)) for x in timestamp_list]

start_timestamp = min(timestamp_list)
timestamp_list = [x - start_timestamp for x in timestamp_list]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plt.xlabel("seconds since start of the load test")
plt.ylabel("QPS")
l1, = ax.plot(timestamp_list, qps_list, label="QPS")
ax2 = ax.twinx()
l2, = ax2.plot(timestamp_list, latency_p50_list, 'C1', label="latency P50 (ms)")
l2, = ax2.plot(timestamp_list, latency_p90_list, 'C2', label="latency P90 (ms)")
l2, = ax2.plot(timestamp_list, latency_p99_list, 'C3', label="latency P99 (ms)")

ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', borderaxespad=0.)
ax2.legend(bbox_to_anchor=(1., 1.02, 1., .102), loc='lower left', borderaxespad=0.)

plt.ylabel("Latency")
plt.title('Per-10-second QPS & Latency')
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### How to interpret the graph above
# MAGIC In the beginning:
# MAGIC * QPS might be __low__, because we started with __fewer__ #parallel-requests
# MAGIC * Latency might be __high__, because it takes time to scale up the provisioned concurrency
# MAGIC 
# MAGIC As we increase #parallel-requests:
# MAGIC * More provisioned concurrency will be added, as a result of scale-up
# MAGIC * QPS will start to ramp up, eventually hitting your target
# MAGIC * Latency will eventually be similar to what we got with the __best run_id__ in __Step 3.2__
