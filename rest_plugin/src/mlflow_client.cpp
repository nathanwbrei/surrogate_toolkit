#include "mlflow_client.h"

uint64_t timeSinceEpochMillisec() {
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

phasm::MLflowRESTClient::MLflowRESTClient(std::string url) {
    baseUrl_ = url;
}

std::string phasm::MLflowRESTClient::SetEndPoint(phasm::MLflowRESTClient::RESTMethods method) {
    switch (method) {
        // Implementation of https://mlflow.org/docs/latest/rest-api.html#get-download-uri-for-modelversion-artifacts
        case phasm::MLflowRESTClient::RESTMethods::GetDownloadUri:
            return "/api/2.0/mlflow/model-versions/get-download-uri";
        // Based on "https://mlflow.org/docs/latest/rest-api.html#log-metric"
        case phasm::MLflowRESTClient::RESTMethods::LogResult:
            return "/api/2.0/mlflow/runs/log-metric";
        // Implementation of https://mlflow.org/docs/latest/rest-api.html#get-registeredmodel
        case phasm::MLflowRESTClient::RESTMethods::GetRegisteredModel:
            return "/api/2.0/mlflow/registered-models/get";
        // Implementation of https://mlflow.org/docs/latest/rest-api.html#list-artifacts
        case phasm::MLflowRESTClient::RESTMethods::ListArtifacts:
        default:
            return "";
    }
}

json::value phasm::MLflowRESTClient::ConvertFieldMapToJSON(const std::map<std::string, std::string>& fieldMap) {
    web::json::value jsonValue;

    for (const auto& pair : fieldMap) {
        jsonValue[pair.first] = web::json::value::string(pair.second);
    }

    return jsonValue;
}

void phasm::MLflowRESTClient::SetRequestBody(const json::value& requestBody){
    request_.set_body(requestBody);
}

void phasm::MLflowRESTClient::SetRequestUri(){
    if (endpoint_.empty()) {
        std::cerr << "Endpoint is empty!\nExit... \n\n";
        exit(-1);
    }
    std::cout << "Sending request to " << baseUrl_ << endpoint_ << std::endl;
    uri_builder builder(endpoint_);
    request_.set_request_uri(builder.to_uri());
}

void phasm::MLflowRESTClient::SetHttpRequest(const json::value& requestBody){
    phasm::MLflowRESTClient::SetRequestUri();
    phasm::MLflowRESTClient::SetRequestBody(requestBody);
}

std::string phasm::MLflowRESTClient::ExtractArtifactUri(const web::json::value& jsonResponse) {
    std::string artifactUri = utility::conversions::to_utf8string(
        jsonResponse.at(utility::string_t("artifact_uri")).as_string());
    return artifactUri;
}

// Somehow I think MLflow logging is not the best avenue for us unless we still need MLflow to do some future
// analysis. MLflow's log matric/batch API must provide a @param run_id, which means stuff are grouped and placed
// by run_ids. I think we need more flexibility on this, maybe a general DB.
// API param list: https://mlflow.org/docs/latest/rest-api.html#log-metric
//
// Test curl command:
// curl -H 'Content-Type: application/json' -X POST http://127.0.0.1:5000/api/2.0/mlflow/runs/log-metric \
// -d '{"run_id": "f136c5849c0a48faab570cc95c81b383", "key": "demo-result", "value": -1.00, "timestamp": 1692736539}'
// Response is an empty json object
bool phasm::MLflowRESTClient::PostResult(std::string run_id, std::string key, double value) {

    // Create an HTTP client object
    http_client client(baseUrl_);
    endpoint_ = phasm::MLflowRESTClient::SetEndPoint(phasm::MLflowRESTClient::RESTMethods::LogResult);
    
    request_.set_method(methods::POST);

    // Create the request body according to required fields
    json::value requestBody;
    requestBody["run_id"] = json::value::string(run_id); // TODO: json::value::value(run_id) instead?
    // Each key has its independent stored file. Each row of this file stands for timestamp, value, and step
    requestBody["key"] = json::value::string(key);
    // cpprestsdk web::json::value class: https://microsoft.github.io/cpprestsdk/classweb_1_1json_1_1value.html
    requestBody["value"] = json::value::number(value); // TODO: json::value::value(run_id) instead?
    requestBody["timestamp"] = json::value::number(timeSinceEpochMillisec());
    std::cout << "Logging metric: [" << key << "] with value: [" << value << "]\n";

    phasm::MLflowRESTClient::SetHttpRequest(requestBody);

    pplx::task<http_response> response = client.request(request_);

    // Get the response status.
    http_response httpResponse = response.get();
    web::http::status_code statusCode = httpResponse.status_code();

    if (statusCode != status_codes::OK) {
        throw std::runtime_error("Logging metric received non-OK status code: " + std::to_string(statusCode));
    }

    return (statusCode == status_codes::OK);
}

std::string phasm::MLflowRESTClient::GetModelDownloadUri(std::string name, std::string version) {
    std::string artifactUri;

    // Create an HTTP client object
    http_client client(baseUrl_);
    endpoint_ = phasm::MLflowRESTClient::SetEndPoint(phasm::MLflowRESTClient::RESTMethods::GetDownloadUri);

    // Create a GET request
    request_.set_method(methods::GET);

    // Create the request body
    json::value requestBody;
    requestBody["name"] = json::value::string(name);
    requestBody["version"] = json::value::string(version);
    phasm::MLflowRESTClient::SetHttpRequest(requestBody);

    // Send the GET request asynchronously
    client.request(request_).then([](http_response response) {
        std::cout << "\nResponse code: " << response.status_code() << std::endl;
        // Check the status code
        if (response.status_code() == status_codes::OK) {
            // Read the response body as a JSON object
            return response.extract_json();
        } else {
            // Handle the error condition
            throw std::runtime_error("Received non-OK status code: " + std::to_string(response.status_code()));
        }
    }).then([this, &artifactUri](web::json::value jsonResponse) {
        // Process the JSON response
        std::cout << "\nFull response:\n" << jsonResponse << std::endl;

        artifactUri = this->ExtractArtifactUri(jsonResponse);
        std::cout << "\nGet artifact_uri from response:\n" << artifactUri << "\n\n";

    }).wait();

    return artifactUri;
}
