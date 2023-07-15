/** 
Copyright 2021, Jefferson Science Associates, LLC.
Subject to the terms in the LICENSE file found in the top-level directory.

First developed by xmei@jlab.org.

References:
- http://www.atakansarioglu.com/easy-quick-start-cplusplus-rest-client-example-cpprest-tutorial/
- https://github.com/microsoft/cpprestsdk
- https://mlflow.org/docs/latest/rest-api.html
*/

#include <cpprest/http_client.h>
#include <cpprest/filestream.h>
// #include <torch/torch.h>

// #include "torch_utils.h"
// #include "torchscript_model.h"

using namespace web;
using namespace web::http;
using namespace web::http::client;

#define LOCAL_HOST U("http://127.0.0.1")
#define DOCKER_HOST U("http://host.docker.internal:5000")
#define ENDPOINT U("/api/2.0/mlflow/registered-models/get")

#define RE_KEY "name"
#define RE_VALUE "demo-reg-model"

std::string extractSourceFromJson(const web::json::value jsonResponse) {

    // Extract the source value from latest_versions
    const auto& latestVersions = jsonResponse.at(U("registered_model")).at(U("latest_versions")).as_array();
    const utility::string_t& source = latestVersions.begin()->at(U("source")).as_string();

    // Convert the source value to std::string
    std::string sourceString = utility::conversions::to_utf8string(source);
    std::string modelPath = sourceString.append("/data/model.pth");

    return modelPath;
}

int main() {
    // Create an HTTP client object
    http_client client(DOCKER_HOST);

    // Create a GET request
    uri_builder builder(ENDPOINT);
    http_request request(methods::GET);
    request.set_request_uri(builder.to_uri());

    // Create the request body
    json::value requestBody;
    requestBody[U(RE_KEY)] = json::value::string(U(RE_VALUE));

    // Set the request body
    request.set_body(requestBody);

    // Send the GET request asynchronously
    std::cout << "Send request" << "..." << std::endl;
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
    }).then([](web::json::value jsonValue) {
        // Process the JSON response
        std::cout << "\nFull response:\n" << jsonValue << std::endl;

        std::string modelPath = extractSourceFromJson(jsonValue);
        std::cout << "\nGet file source from response:\n" << modelPath << std::endl;

        // phasm::TorchscriptModel model = phasm::TorchscriptModel(modelPath, true);

    }).wait();

    return 0;
}
