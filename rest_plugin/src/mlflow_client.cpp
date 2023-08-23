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
        // Based on https://mlflow.org/docs/latest/rest-api.html#log-metric
        case phasm::MLflowRESTClient::RESTMethods::LogResult:
            return "/api/2.0/mlflow/runs/log-metric";
        // Based on https://mlflow.org/docs/latest/rest-api.html#get-modelversion
        case phasm::MLflowRESTClient::RESTMethods::GetRunID:
            return "/api/2.0/mlflow/model-versions/get";
        // Implementation of https://mlflow.org/docs/latest/rest-api.html#list-artifacts
        case phasm::MLflowRESTClient::RESTMethods::ListArtifacts:
            return "/api/2.0/mlflow/artifacts/list";
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

void phasm::MLflowRESTClient::SetRequestUri(const phasm::MLflowRESTClient::RESTMethods& restMethod){
    endpoint_ = phasm::MLflowRESTClient::SetEndPoint(restMethod);
    if (endpoint_.empty()) {
        std::cerr << "Endpoint is empty!\nExit... \n\n";
        exit(-1);
    }
    std::cout << "Sending request to " << baseUrl_ << endpoint_ << std::endl;
    uri_builder builder(endpoint_);
    request_.set_request_uri(builder.to_uri());
}

void phasm::MLflowRESTClient::SetHttpRequest(
    const http::method& httpMethod,
    const json::value& requestBody,
    const phasm::MLflowRESTClient::RESTMethods& restMethod
    ){
    phasm::MLflowRESTClient::SetRequestUri(restMethod);
    request_.set_method(httpMethod);
    request_.set_body(requestBody);
}

std::string phasm::MLflowRESTClient::ExtractArtifactUri(const web::json::value& jsonResponse) {
    if (!jsonResponse.has_field(utility::string_t("artifact_uri")))
        return "";

    return utility::conversions::to_utf8string(jsonResponse.at(utility::string_t("artifact_uri")).as_string());
}

std::string phasm::MLflowRESTClient::ExtractRunID(const web::json::value& jsonResponse) {
    if (!jsonResponse.has_field(utility::string_t("model_version")))
        return "";

    auto context = jsonResponse.at(utility::string_t("model_version"));
    if (!context.has_field(utility::string_t("run_id")))
        return "";

    return context.at(utility::string_t("run_id")).as_string();
}

/* Curl command:
curl -H 'Content-Type: application/json' -X GET http://127.0.0.1:5004/api/2.0/mlflow/model-versions/get\
-d '{"name": "demo-reg-model", "version": "1"}'
Full response:
{
  "model_version": {
    "name": "demo-reg-model",
    "version": "1",
    "creation_timestamp": 1692759380663,
    "last_updated_timestamp": 1692759380663,
    "current_stage": "None",
    "source": "file:///Users/xinxinmei/Documents/projects/phasm/examples/mlflow_rest/demo_mlflow_host/mlruns/0/fab5f88fdbef4a1dbdc036c93e4a61b5/artifacts/model",
    "run_id": "fab5f88fdbef4a1dbdc036c93e4a61b5",
    "status": "READY"
  }
*/
std::string phasm::MLflowRESTClient::GetRunID(std::string name, std::string version) {
    std::string run_id;

    // Create the request body
    json::value requestBody;
    requestBody["name"] = json::value::string(name);
    requestBody["version"] = json::value::string(version);

    // Create the http client.
    // phasm::MLflowRESTClient::SetHttpRequest depends on baseUrl_, so this sentence must come first.
    http_client client(baseUrl_);
    phasm::MLflowRESTClient::SetHttpRequest(methods::GET, requestBody, phasm::MLflowRESTClient::RESTMethods::GetRunID);

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
    }).then([this, &run_id](web::json::value jsonResponse) {
        // Process the JSON response
        // std::cout << "\nFull response:\n" << jsonResponse << std::endl;

        run_id = this->ExtractRunID(jsonResponse);
        std::cout << "\nGet run_id from response:\n" << run_id << "\n\n";

    }).wait();

    return run_id;
}


// Somehow I think MLflow logging is not the best avenue for us unless we still need MLflow to do some future
// analysis. MLflow's log matric/batch API must provide a @param run_id, which means stuff are grouped and placed
// by run_ids. I think we need more flexibility on this, maybe a general DB.
// API param list: https://mlflow.org/docs/latest/rest-api.html#log-metric
//
/* Curl command:
curl -H 'Content-Type: application/json' -X POST http://127.0.0.1:5004/api/2.0/mlflow/runs/log-metric \
-d '{"run_id": "f136c5849c0a48faab570cc95c81b383", "key": "demo-result", "value": -1.00, "timestamp": 1692736539}'
Response is an empty json object
*/
bool phasm::MLflowRESTClient::PostResult(std::string run_id, std::string key, double value) {
    // Create the request body according to required fields
    json::value requestBody;
    requestBody["run_id"] = json::value::string(run_id);
    // Each key has its independent stored file. Each row of this file stands for timestamp, value, and step
    requestBody["key"] = json::value::string(key);
    // cpprestsdk web::json::value class: https://microsoft.github.io/cpprestsdk/classweb_1_1json_1_1value.html
    requestBody["value"] = json::value::number(value); // TODO: json::value::value(run_id) instead?
    requestBody["timestamp"] = json::value::number(timeSinceEpochMillisec());
    std::cout << "\nLogging metric: [" << key << "] with value: [" << value << "]\n\n";

    http_client client(baseUrl_);
    phasm::MLflowRESTClient::SetHttpRequest(methods::POST, requestBody, phasm::MLflowRESTClient::RESTMethods::LogResult);

    pplx::task<http_response> response = client.request(request_);

    // Get the response status.
    http_response httpResponse = response.get();
    web::http::status_code statusCode = httpResponse.status_code();

    if (statusCode != status_codes::OK) {
        throw std::runtime_error("Logging metric received non-OK status code: " + std::to_string(statusCode));
    }

    return (statusCode == status_codes::OK);
}


// Param list: https://mlflow.org/docs/latest/rest-api.html#list-artifacts
// Here we do not consider page_token or next_page_token.
//
/* Curl coommand:
curl -H 'Content-Type: application/json' -X GET http://127.0.0.1:500 -d '{"run_id": "fab5f88fdbef4a1dbdc036c93e4a61b5"}'
Full response:
{
  "root_uri": "file:///Users/xinxinmei/Documents/projects/phasm/examples/mlflow_rest/demo_mlflow_host/mlruns/0/fab5f88fdbef4a1dbdc036c93e4a61b5/artifacts",
  "files": [
    {
      "path": "model",
      "is_dir": true
    }
  ]
}
*/
web::json::value phasm::MLflowRESTClient::ListArtifacts(std::string run_id) {
    web::json::value ret = web::json::value();

    json::value requestBody;
    requestBody["run_id"] = json::value::string(run_id);

    http_client client(baseUrl_);
    phasm::MLflowRESTClient::SetHttpRequest(methods::GET, requestBody, phasm::MLflowRESTClient::RESTMethods::ListArtifacts);

    pplx::task<http_response> response = client.request(request_);

    // Wait for the response and extract the JSON value
    http_response httpResponse = response.get();
    web::http::status_code statusCode = httpResponse.status_code();

    if (statusCode != status_codes::OK) {
        throw std::runtime_error("List Artifacts reequest received non-OK status code: " + std::to_string(statusCode));
    }

    return httpResponse.extract_json().get(); //.get() is necessary for type conversion. It's similar to adding a .wait()
}

std::string phasm::MLflowRESTClient::GetModelDownloadUri(std::string name, std::string version) {
    std::string artifactUri;

    // Create the request body
    json::value requestBody;
    requestBody["name"] = json::value::string(name);
    requestBody["version"] = json::value::string(version);

    http_client client(baseUrl_);
    phasm::MLflowRESTClient::SetHttpRequest(methods::GET, requestBody, phasm::MLflowRESTClient::RESTMethods::GetDownloadUri);

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
        // std::cout << "\nFull response:\n" << jsonResponse << std::endl;

        artifactUri = this->ExtractArtifactUri(jsonResponse);
        std::cout << "\nGet artifact_uri from response:\n" << artifactUri << "\n\n";

    }).wait();

    return artifactUri;
}
