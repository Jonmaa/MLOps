# MLOps

This GitHub repository tests different well-known MLOps tools with various artificial intelligence models. At the same time, a GitHub Actions pipeline will be created for them to see what the process would be like for deploying new models.

## 1. Installing Required Dependencies

Each tool has its own requirements.txt file within its folder to install all the necessary dependencies for execution.

```bash
pip install -r requirements.txt
```

## 2. MLflow

### 2.1 MNIST

```bash
python mlflowMain.py
```

```bash
python mlflowTest.py
```

### 2.2 Sentiment

```bash
python mlflowMain_sentiment.py
```

```bash
python mlflowTest_sentiment.py
```

With the following command you can access the MLflow user interface to view different executions. It works as a logger, allowing easy comparison of differences in execution results with different metrics. This way, you can determine which model and parameters give the best result for the desired action, in this case classifying MNIST digits.

```bash
mlflow ui
```

## 3. Metaflow

### 3.1 MNIST

```bash
python metaflowMain.py run
```

### 3.2 Sentiment

```bash
python metaflowMain_sentiment.py run
```

>[!NOTE]
You can copy these files to the Metaflow sandbox and execute them. Once execution is finished, a URL will appear in the terminal that you can access to view the results visually.

## 4. Apache Airflow

### 4.1 Steps to Execute
- Create a folder to install Airflow
- Generate an environment in that folder
- Activate the environment
- Follow the installation steps from the official page, which are 4-5 commands

With the following command the UI is executed and can be viewed at localhost:8080

```bash
airflow standalone
```
When doing this, a series of files will be generated in the folder where we installed it. If we want to change the Airflow configuration, edit airflow.cfg

Workflows in Airflow are called DAGs, which is why we will create a new folder called dags in the directory, where we will add our files.
We can create the first workflow there and reload the page to see it. If it doesn't appear, press Ctrl + C to stop airflow standalone and run it again.

Once on the page we can execute our file by clicking a button and see how the different steps are performed.

>[!IMPORTANT]
To be able to execute the files in the UI, it is necessary to have installed the dependencies (requirements.txt) beforehand in a custom environment.

## 5. Kubeflow Pipelines

### 5.1 Steps to Execute
- Install Minikube and start it with minimum resources for Kubeflow

```bash
minikube start --cpus=4 --memory=8192
```
- Install Kubeflow in Minikube
```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=1.8.0"
```
```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=1.8.0"
```
- Wait for all Kubeflow pods to be running, which can be checked with the following command
```bash
kubectl get pods -n kubeflow
```
- Once all are running, you can do port-forwarding to open the user interface
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
- Now you can access it through localhost:8080
- To perform a run in Kubeflow Pipelines, first you need to import the YAML, for which you execute the kubeflowMain.py file
- This will generate a .yaml file, which is what gets uploaded in the Kubeflow pipelines section by clicking select file.
- Finally, go to the runs section, create a new one with the recently imported pipeline and give the execution a name. The moment you save it, you can visualize how it executes.

>[!IMPORTANT]
To be able to execute the files in the UI, it is necessary to have installed the dependencies (requirements.txt) beforehand in a custom environment.

## Pipeline

When making any commit or PR to the repository, main.yml will be executed, which will check what files have changed and only execute the jobs related to those files.