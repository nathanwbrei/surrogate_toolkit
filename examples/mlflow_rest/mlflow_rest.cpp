/** 
Copyright 2023, Jefferson Science Associates, LLC.
Subject to the terms in the LICENSE file found in the top-level directory.

First developed by xmei@jlab.org.

References:
- http://www.atakansarioglu.com/easy-quick-start-cplusplus-rest-client-example-cpprest-tutorial/
- https://github.com/microsoft/cpprestsdk
- https://mlflow.org/docs/latest/rest-api.html
*/

// Put torch headers on top to avoid conflicts!!! Serious errors awaiting!
#include "torchscript_model.h"

#include "../../rest_plugin/include/mlflow_client.h"

#include <filesystem>

/**
 * This method is designed particularly for local models (in the same folder) host outside of docker image.
 * It uses string methods to search some keywords in @param uri and operate strings.
 * @todo(cissie) The method is not appiable for other situations.
*/
std::string getLocalModelPathByArtifactUri(const std::string& uri) {
    std::filesystem::path filePath(__FILE__);
    std::string pathPrefix = filePath.parent_path().string(); // this file's parent path

    std::string currentFolderName = filePath.parent_path().filename().string();
    size_t pos = uri.find(currentFolderName);

    if (pos == std::string::npos) {
        return "";  // error hanlding
    }

    std::string path_suffix = uri.substr(pos + currentFolderName.length());
    return pathPrefix + path_suffix + "/data/model.pth";
}

int main() {

    phasm::MLflowRESTClient client("http://host.docker.internal:5004"); // local host address inside the docker image

    // Make sure on the MLFLow side, a model was registered as "demo-reg-model".
    std::string uri = client.GetModelDownloadUri("demo-reg-model", "1");
    if (uri.empty()) {
        std::cerr << "Error: empty artifact uri. \n\nExit...\n\n";
        exit(-1);
    }

    std::string modelPath = getLocalModelPathByArtifactUri(uri);
    if (modelPath == ""){
        std::cerr << "Error: model does not exist. \n\nExit...\n\n";
        exit(-1);
    }
    std::cout << "Extracted model path:\n" << modelPath << "\n\n";

    std::string run_id = client.GetRunID("demo-reg-model", "1");
    if (run_id == ""){
        std::cerr << "Error: empty run_id. \n\nExit...\n\n";
        exit(-1);
    }

    // Model inference
    phasm::TorchscriptModel model = phasm::TorchscriptModel(modelPath);
    // std::vector<int64_t> first_layer_shape = model.GetFirstLayerShape();
    // std::cout << first_layer_shape << std::endl; // 30, 4

    // Check inference result. Using the same test point with Python script.
    std::vector<torch::jit::IValue> input;
    float testDatapoint[] = {4.4000, 3.0000, 1.3000, 0.2000};
    input.push_back(torch::from_blob(testDatapoint, {4}));
    auto output = model.get_module().forward(input).toTensor();
    torch::Tensor argmaxIndices = at::argmax(output);
    std::cout << "Prediction of [4.4000, 3.0000, 1.3000, 0.2000] is \n[" << output << "\n]\n";
    std::cout << "Argmax Indices:\n" << argmaxIndices << "\n";

    std::string postStatus = client.PostResult(run_id, "Argmax_index", 0) ? "Succeed" : "Failed";
    std::cout << "Posting argmax_index to server..." << postStatus << std::endl << std::endl;

    // Verify argmax index.
    // Python output > Argmax value: 0
    if (argmaxIndices.item<int>() > 0) {
        std::cout << "\nPrediction error: argmax(output) != 0!" << std::endl;
        return -1;
    }
    std::cout << "\nPrediction result is the same with Python!" << std::endl;
    return 0;
}
