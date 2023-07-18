MLFlow Steps
-------------

1. Launch a new run
     mlflow run . --env-manager=local
  A registered model named "demo-reg-model" is 
2. Launch a local mlflow server via ``mlflow ui --dev`` (debug mode), which will open the local host at ``http://127.0.0.1:5000``.
3. Make a GET request via ``http://127.0.0.1:5000/api/2.0/mlflow/registered-models/get?name=demo-reg-model``.
4. To make a request inside the docker, launch the docker with
    docker --network host
