### Steps to run the example

1. (optinal) Investigate to `demo_mlflow_host` repo, Start a local run. This will register a model called "demo-reg-model".
    ```
    mlflow run . --env-manager=local
    ```
2. Start the server.
    ```
    mlflow ui --dev
    ```
3. Run Postman or phasm-example-rest to connect to this server. If you launch the example inside the container, run the container with `--network host` option
    ```
    docker run --it --network host -v ${PWD}:/app <docker_image_id>
    ```
