# MLOps

En este repositorio de GitHub se van a probar diferentes herramientas conocidas de MLOps con distintos modelos de inteligencia artificial. Al mismo tiempo se va a crear un pipeline de Github Actions para los mismos y ver como sería el proceso para desplegar nuevos modelos.

## 1. Instalación de las dependencias necesarias

```bash
pip install -r requirements.txt
```

## 2. Mlflow
Intento de implementación de un modelo de IA clasificador de MNIST. Para generar el modelo es necesario ejecutar el main y para probarlo el test.

```bash
python mlflowMain.py
```

```bash
python mlflowTest.py
```
Con el siguiente comando se podrá acceder a la interfaz de usuario de mlflow para ver las diferentes ejecuciones, funciona como un logger, de forma que permite comparar fácilmente las diferencias en los resultados de las ejecuciones con distintas métricas. Así, se puede saber que modelo y con que parámetros da el mejor resultado para la acción que se quiera realizar, en este caso clasificar los digitos MNIST.

```bash
python mlflow ui
```

## 3. Metaflow
Otro intento de implementación de un modelo de IA clasificador de MNIST. Metaflow solo se puede ejecutar en linux y mac, en mi caso probado con ubuntu. Actualmente no se puede instalar en Windows a menos que se haga uso del WSL.

Metaflow se basa en la división por pasos a la hora de realizar el código, teniendo que definir los diferentes pasos/steps e indicando en cada uno de ellos cual es el siguiente que debe realizar. 

Actualmente tiene en desarrollo una interfaz de usuario, no aparece nada de información al respecto en la documentación de la página oficial a día de hoy. Pero se puede buscar en los diferentes repositorios de github de los desarrolladores para intentar instalarlo. Lo he intentando pero a la hora de desplegarlo me da un error debido a que no recibe los datos a la hora de intentar realizar una ejecución del código. Es por ello que para probarlo se ha hecho uso de https://docs.outerbounds.com/sandbox/ el sandbox que tiene, que si le da por ejecutar, se puede copiar el código en uno de los archivos y a la hora de ejecutarlo saldrá por la línea de comandos una url a la que se podrá acceder para visualizar la UI.

```bash
python metaflowMain.py run
```

## 4. Apache airflow
En este caso se implementa un modelo de IA que se encarga de analizar si las reviews de unos usuarios sobre películas son positivas o negativas. Se hace uso del conjunto de datos de IMDB de Hugging Face y para analizar los sentimientos se hace uso de un modelo DistilBERT pre-entrenado para analizar los sentimientos en reseñas.

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
Para poder ejecutar airflowMain.py es necesario instalar las siguientes dependencias: pip install datasets torch transformers

## 5. Kubeflow pipelines
En este caso se implementa un modelo de IA que se encarga de analizar si las reviews de unos usuarios sobre películas son positivas o negativas. Se hace uso del conjunto de datos de IMDB de Hugging Face y para analizar los sentimientos se hace uso de un modelo DistilBERT pre-entrenado para analizar los sentimientos en reseñas.

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
Para poder ejecutar kubeflowMain.py es necesario instalar las siguientes dependencias (se recomienda que se instalen en un env personalizado): pip install kfp==1.8.9 datasets pandas transformers torch scikit-learn

## Pipeline

Al hacer cualquier cualquier commit o pr al repositorio se ejecutará el main.yml para comprobar que todo funciona correctamente. Además este comprobará la accuracy del nuevo modelo y si es mayor a la indicada, se subira y se podrá ver en la parte de artifacts.
