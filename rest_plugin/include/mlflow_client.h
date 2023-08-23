// Copyright 2023, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

// A Python reference: https://github.com/mlflow/mlflow/blob/master/examples/rest_api/mlflow_tracking_rest_api.py
// Full MLFlow REST APIs: https://mlflow.org/docs/latest/rest-api.html

#include <cpprest/http_client.h>
#include <cpprest/filestream.h>
#include <chrono>

using namespace web;
using namespace web::http;
using namespace web::http::client;
using namespace std::chrono;

namespace phasm {

    class MLflowRESTClient {

        public:

            /// @brief Constructor. Construct a new MLflowRESTClient object accoring to @param url.
            /// @param url the base url of the MLFLow server.
            MLflowRESTClient(std::string url);

            /// @brief Implementation of "Get Download URI For ModelVersion Artifacts" at
            /// https://mlflow.org/docs/latest/rest-api.html#get-download-uri-for-modelversion-artifacts.
            /// Get the model artifact download uri according to the registered model name and version.
            /// @param name name of the registered model.
            /// @param version version of the registered model. 
            /// @return uri of the model artifact.
            std::string GetModelDownloadUri(std::string name, std::string version);

            /// @brief Get run_id of a registered model given its name and version.
            /// Based on https://mlflow.org/docs/latest/rest-api.html#get-modelversion
            /// @param name name of the registered model.
            /// @param version version of the registered model.
            /// @return model's run_id.
            std::string GetRunID(std::string name, std::string version);

            /// @breif Using the "Log Metric" API to log a datapoint to MLFlow server.
            /// https://mlflow.org/docs/latest/rest-api.html#log-metric
            /// @param run_id an identifier of the model run.
            /// @param key the name of the logged data.
            /// @param value the value of the looged data.
            bool PostResult(std::string run_id, std::string key, double value);

        private:
            std::string baseUrl_ = "";
            std::string endpoint_ = "";
            http_request request_;

            enum class RESTMethods {
                GetDownloadUri,
                LogResult,
                GetRunID,
                ListArtifacts
            };

            /// @brief Set endpoint according to defined methods.
            /// @param method is one of the methods listed in the enum class RESTMethods.
            std::string SetEndPoint(RESTMethods method);

            void SetHttpRequest(const http::method& httpMethod, const json::value& requestBody, const phasm::MLflowRESTClient::RESTMethods& restMethod);

            /// @brief A helper fuction to set and print the api url.
            /// @param method is one of the methods listed in the enum class RESTMethods.
            void SetRequestUri(const phasm::MLflowRESTClient::RESTMethods& method);

            /// @brief For GetDownloadUri method, extract the uri from the origin json response.
            /// @param jsonResponse origin json response.
            /// @return uri of the model artifact.
            std::string ExtractArtifactUri(const web::json::value& jsonResponse);

            /// @brief For GetRunID method, extract the run_id from the origin json response.
            /// @param jsonResponse origin json response.
            /// @return run_id of the model.
            std::string ExtractRunID(const web::json::value& jsonResponse);

            json::value ConvertFieldMapToJSON(const std::map<std::string, std::string>& fieldMap);
    };
}
