/** 
Copyright 2021, Jefferson Science Associates, LLC.
Subject to the terms in the LICENSE file found in the top-level directory.

First developed by xmei@jlab.org.

References:
- http://www.atakansarioglu.com/easy-quick-start-cplusplus-rest-client-example-cpprest-tutorial/
- https://github.com/microsoft/cpprestsdk
- https://mlflow.org/docs/latest/rest-api.html
*/

// Put torch headers on top to avoid conflicts!!! Serious errors awaiting!
#include "torchscript_model.h"

#include <filesystem>
#include <cpprest/http_client.h>
#include <cpprest/filestream.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;

std::string extractArtifactUriFromJsonResponseString(const web::json::value jsonResponse) {
    std::string artifactUri = utility::conversions::to_utf8string(
        jsonResponse.at(utility::string_t("artifact_uri")).as_string());
    return artifactUri;
}

/**
 * This method is designed particularly for local models (mlflow host in the same folder).
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
    utility::string_t LOCAL_HOST("http://127.0.0.1");
    utility::string_t DOCKER_HOST("http://host.docker.internal:5000");

    // API: https://mlflow.org/docs/latest/rest-api.html#get-download-uri-for-modelversion-artifacts
    utility::string_t ENDPOINT("/api/2.0/mlflow/model-versions/get-download-uri");
    utility::string_t REQ_KEY_1("name");
    utility::string_t REQ_VALUE_1("demo-reg-model");
    utility::string_t REQ_KEY_2("version");
    utility::string_t REQ_VALUE_2("1");

    // Create an HTTP client object
    http_client client(DOCKER_HOST);

    // Create a GET request
    uri_builder builder(ENDPOINT);
    http_request request(methods::GET);
    request.set_request_uri(builder.to_uri());

    // Create the request body
    json::value requestBody;
    requestBody[REQ_KEY_1] = json::value::string(REQ_VALUE_1);
    requestBody[REQ_KEY_2] = json::value::string(REQ_VALUE_2);

    // Set the request body
    request.set_body(requestBody);

    // Send the GET request asynchronously
    std::cout << "Sending request to " << ENDPOINT << std::endl;
    client.request(request).then([](http_response response) {
        std::cout << response.status_code() << std::endl;
        // Check the status code
        if (response.status_code() == status_codes::OK) {
            // Read the response body as a JSON object
            return response.extract_json();
        } else {
            // Handle the error condition
            throw std::runtime_error("Received non-OK status code: " + std::to_string(response.status_code()));
        }
    }).then([](web::json::value jsonResponse) {
        // Process the JSON response
        std::cout << "\nFull response:\n" << jsonResponse << std::endl;

        std::string artifactUri = extractArtifactUriFromJsonResponseString(jsonResponse);
        std::cout << "\nGet artifact_uri from response:\n" << artifactUri << "\n\n";

        std::string modelPath = getLocalModelPathByArtifactUri(artifactUri);
        if (modelPath == ""){
            std::cout << "Error: model does not exist. \n\nExit -1...\n\n";
            exit(-1);
        }
        std::cout << "Extracted model path:\n" << modelPath << "\n\n";

        // Model inference
        phasm::TorchscriptModel model = phasm::TorchscriptModel(modelPath);
        // std::vector<int64_t> first_layer_shape = model.GetFirstLayerShape();
        // std::cout << first_layer_shape << std::endl; // 30, 4

        std::vector<torch::jit::IValue> input;
        float testDatapoint[] = {4.4000, 3.0000, 1.3000, 0.2000};
        input.push_back(torch::from_blob(testDatapoint, {4}));
        auto output = model.get_module().forward(input).toTensor();
        std::cout << "Prediction of [4.4000, 3.0000, 1.3000, 0.2000] is \n[" << output << "\n]\n";
        std::cout << "Python value: [  9.0987,   3.6623, -10.0076]\n\n";

    }).wait();

    return 0;
}
