MLFlow Steps
-------------
0. Include `/install/plugins` `/install/lib` to `LD_LIBRARY_PATH`.
1. Delete existing `mlruns` folder. Launch a new run
     mlflow run . --env-manager=local
  A model named "demo-reg-model" is registered.
2. Launch a local mlflow server via ``mlflow ui --port 5004 --dev`` (debug mode), which will open the local host at ``http://127.0.0.1:5004``.
3. Make a GET request via ``http://127.0.0.1:5000/api/2.0/mlflow/registered-models/get?name=demo-reg-model``.
4. To make a request inside the docker, launch the docker with option `--network host`.
