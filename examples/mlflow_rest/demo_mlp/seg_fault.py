
"""
First checked in by xmei@jlab.org.

- Modified based on the MLFlow PyTorch TorchScript example:
https://github.com/mlflow/mlflow/tree/master/examples/pytorch/torchscript/IrisClassification
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.models import infer_signature


class SimpleMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = self.out(x)
        return x


def prepare_data():
    training_data_filename = "training_captures.csv"

    print("Loading training data from '" + training_data_filename + "'")
    df = pd.read_csv(training_data_filename)

    features = df[['x',' y',' z']].values
    targets = df[[' Bx',' By',' Bz']].values
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    return X_train, X_test, y_train, y_test


def train_model(model, epochs, X_train, y_train):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        out = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step = epoch + 1
        if step % 10 == 0:
            print("step", step, ", loss", float(loss))

    return model


def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predict_out = model(X_test)
        loss_fn = nn.MSELoss()
        print("\nTest loss: ", loss_fn(predict_out, y_test).item())
        return infer_signature(X_test.numpy(), predict_out.numpy())


def refresh_folder(ab_path):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified mfield MLP Torchscripted model")

    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of epochs to run (default: 1000)"
    )

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data()
    model = SimpleMLP(3, 30, 3)
    model = torch.jit.script(model)

    # print("\nModel before training\n", model.modules)
    model = train_model(model, args.epochs, X_train, y_train)
    # print("\nModel after training\n", model.modules)

    signature = test_model(model, X_test, y_test)

    model_name = "simpleMLP.pth"
    model.save(model_name)
    loaded_model = torch.jit.load(model_name)

    print(f"\n\nLoad model {model_name} in current dir")
    loaded_model.eval()
    with torch.no_grad():
        test_datapoint = torch.Tensor([0.0, 0.0, 0.0])
        prediction = loaded_model(test_datapoint)
        print("\nPREDICTION RESULT: {}".format(prediction))

    with mlflow.start_run() as run:
        artifact_name = "model"   # foldername-alike
        model_path = mlflow.get_artifact_uri()
        print(f"{model_path}, {run.info.run_id}")

        # TODO: ERROR mlflow.cli: === Run (ID '<id>') failed ===
        # save_model locally scucceed with segfault 11.
        # log_model did not succeed.
        # registered_model_name="simpleMLP"
        mlflow.pytorch.save_model(
            model, artifact_name, signature=signature
        )

    # print("run_id: {}".format(run.info.run_id))
    # for artifact_path in ["model/data"]:
    #     artifacts = [
    #         f.path for f in MlflowClient().list_artifacts(run.info.run_id, artifact_path)
    #     ]
    #     print("artifacts: {}".format(artifacts))
   