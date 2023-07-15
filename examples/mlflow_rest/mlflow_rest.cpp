/** 
Copyright 2021, Jefferson Science Associates, LLC.
Subject to the terms in the LICENSE file found in the top-level directory.

First developed by xmei@jlab.org.

Draft version copied from ChatGPT: https://chat.openai.com/share/a989a968-2ce2-47f8-a398-0329cb35b60c.

References:
- http://www.atakansarioglu.com/easy-quick-start-cplusplus-rest-client-example-cpprest-tutorial/
- https://github.com/microsoft/cpprestsdk
*/

#include <cpprest/http_client.h>
#include <cpprest/filestream.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;

#define LOCAL_HOST U("http://127.0.0.1")
#define DOCKER_HOST U("http://host.docker.internal:5000")
#define ENDPOINT U("/api/2.0/mlflow/registered-models/get")

#define RE_KEY "name"
#define RE_VALUE "demo-reg-model"

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
    std::cout << requestBody << std::endl;

    // Set the request body
    request.set_body(requestBody);

    // Send the GET request asynchronously
    client.request(request).then([](http_response response) {
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
        // Here, you can access the response data using the jsonValue object

        // For example, print the response body
        std::cout << jsonValue.serialize() << std::endl;
    }).wait();

    return 0;
}
