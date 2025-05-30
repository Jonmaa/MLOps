# MLOps

En este repositorio de GitHub se van a probar diferentes herramientas conocidas de MLOps con distintos modelos de inteligencia artificial. Al mismo tiempo se va a crear un pipeline de Github Actions para los mismos y ver como sería el proceso para desplegar nuevos modelos.

## 1. Instalación de las dependencias necesarias

Cada una de las herramientas tiene su archivo requirements.txt dentro de su carpeta para poder instalar todas las dependencias necesarias para su ejecución.

```bash
pip install -r requirements.txt
```

## 2. Mlflow

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

Con el siguiente comando se podrá acceder a la interfaz de usuario de mlflow para ver las diferentes ejecuciones, funciona como un logger, de forma que permite comparar fácilmente las diferencias en los resultados de las ejecuciones con distintas métricas. Así, se puede saber que modelo y con que parámetros da el mejor resultado para la acción que se quiera realizar, en este caso clasificar los digitos MNIST.

```bash
python mlflow ui
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
Se pueden copiar estos archivos en el sandbox de metaflow y ejecutarlos. Una vez terminada la ejecución por la terminal aparecerá una url a la que se podrá acceder para ver los resultados de forma visual.

## 4. Apache airflow

### 4.1 Pasos a seguir para ejecutar
- Crear una carpeta donde instalar airflow
- Generar un environment en esa carpeta
- Activar el environment
- Seguir los pasos de instalación de la página oficial, son 4-5 comandos

Con el siguiente comando se ejecuta la UI y se podrá ver en localhost:8080

```bash
airflow standalone
```
Al hacerlo, en la carpeta donde lo hemos instalado se generaran una serie de ficheros, si queremos cambiar la configuración de airflow editar airflow.cfg

Los workflows en airflow se llaman dags, es por ello que en la carpeta vamos a crear una nueva carpeta llamada dags, donde añadiremos nuestros archivos.
Podemos crear ahí el primer workflow y se puede recargar la página para verlo, en caso de que no se vea, realizar Ctrl + C para parar el airflow standalone y volver a ejecutarlo.

Una vez en la página podremos ejecutar nuestro archivo dándole a un botón y se podrá ver como se van realizando los diferentes pasos.

>[!IMPORTANT]
Para poder ejecutar los archivos en la UI es necesario haber instalado las dependencias (requirements.txt) antes en un environment personalizado.

## 5. Kubeflow pipelines

### 5.1 Pasos a seguir para ejecutar
- Instalar Minikube e iniciarlo con recursos mínimos para kubeflow

```bash
minikube start --cpus=4 --memory=8192
```
- Instalar kubeflow en minikube
```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=1.8.0"
```
```bash
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=1.8.0"
```
- Esperar a que todos los pods de kubeflow se encuentren en ejecución, se pueden mirar con el siguiente comando
```bash
kubectl get pods -n kubeflow
```
- Una vez todos estén en ejecución se puede hacer port-forwarding para abrir la interfaz de usuario
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```
- Ahora se puede acceder a ella a través de localhost:8080
- Para poder realizar una run en kubeflow pipelines, primero se ha de importar el YAML, para ello se ejecuta el archivo de kubeflowMain.py
- Eso generará un .yaml, que será lo se sube en el apartado de pipelines de kubeflow, dándole a seleccionar archivo.
- Por último se va al apartado de runs, se crea una nueva con la pipeline recién importada y se le da un nombre a la ejecución, en el momento en que guarde se podrá visualizar como se ejecuta.

>[!IMPORTANT]
Para poder ejecutar los archivos en la UI es necesario haber instalado las dependencias (requirements.txt) antes en un environment personalizado.


## Pipeline

Al hacer cualquier cualquier commit o pr al repositorio se ejecutará el main.yml que comprobará que archivos han sufrido cambios y solo ejecutar los jobs relacionados con dichos archivos.