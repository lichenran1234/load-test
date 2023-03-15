### Step 1: Install Locust on your own computer

Follow the [official locust docs](https://docs.locust.io/en/stable/installation.html) to install locust on your local machine.

### Step 2: write a locust file `load_test.py` with the load test logics

### Step 3: run the load test

Inside the folder where you have the `load_test.py` file, run the command `locust -f load_test.py`. That will start the locust program and UI. Visit http://0.0.0.0:8089 on your computer and you’ll be able to start the load test. The UI looks like this:
![image1](https://user-images.githubusercontent.com/5786126/222304686-0f9fd25e-d168-4625-8ced-b45b22e697fa.png)