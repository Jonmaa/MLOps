# MLOps

Intento de implementación de un modelo de IA clasificador de MINST con Mlflow y pipeline de Github Actions

## Para poder ejecutar en local

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

```bash
python test.py
```

## Pipeline

Al hacer cualquier cualquier commit o pr al repositorio se ejecutará el main.yml para comprobar que todo funciona correctamente. Además este comprobará la accuracy del nuevo modelo y si es mayor a la indicada, se subira y se podrá ver en la parte de artifacts.