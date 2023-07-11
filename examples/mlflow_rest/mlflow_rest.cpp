/** 
Copyright 2021, Jefferson Science Associates, LLC.
Subject to the terms in the LICENSE file found in the top-level directory.

First developed by xmei@jlab.org.

A toy example to send GET request from REST host http://api.zippopotam.us/us/23606.
Draft version copied from ChatGPT: https://chat.openai.com/share/a989a968-2ce2-47f8-a398-0329cb35b60c.

```
(base) bash-4.4$ curl --location 'http://api.zippopotam.us/us/23606'
Response:
    {
        "post code": "23606",
        "country": "United States",
        "country abbreviation": "US",
        "places": [{
            "place name": "Newport News",
            "longitude": "-76.4967",
            "state": "Virginia",
            "state abbreviation": "VA",
            "latitude": "37.0768"
            }]
    }
```

References:
- http://www.atakansarioglu.com/easy-quick-start-cplusplus-rest-client-example-cpprest-tutorial/
- https://github.com/microsoft/cpprestsdk
*/

#include <cpprest/http_client.h>
#include <cpprest/filestream.h>

using namespace web;
using namespace web::http;
using namespace web::http::client;

int main() {
    // Create an HTTP client object
    http_client client(U("http://api.zippopotam.us"));

    // Create a GET request
    uri_builder builder(U("/us/23606"));
    http_request request(methods::GET);
    request.set_request_uri(builder.to_uri());

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
