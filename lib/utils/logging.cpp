#include "utils/logging.hpp"

std::string utils::logging::generateDateString(){
    // Get the current time as a time_point object
    auto now = std::chrono::system_clock::now();

    // Get the timestamp
    auto timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());

    // Convert the time_point object to a time_t object
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);

    // Convert the time_t object to a tm struct for formatting
    std::tm* localTime = std::localtime(&currentTime);

    // Print the formatted date and time
    std::ostringstream dateStream;
    dateStream << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");
    std::string dateString = dateStream.str();

    return dateString;
}

std::string utils::logging::generateTimestamp(){
    // Get the current time as a time_point object
    auto now = std::chrono::system_clock::now();

    // Get the timestamp
    auto timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count());

    return timestamp;
}

void utils::logging::buildLogFile(std::map<std::string, std::string>& dataMap){
    // Generate the date and timestamp
    std::string dateString = generateDateString();
    std::string timestamp = generateTimestamp();

    // Add the date to the dataMap
    dataMap["date"] = dateString;
    dataMap["timestamp"] = timestamp;

    // Create a map of key-value pairs
    nlohmann::json j = dataMap;

    // Write the nlohmann::json object to a file
    std::string filename = "log_" + timestamp + ".json";
    std::ofstream o("log/" + filename);
    o << std::setw(4) << j << std::endl;  // Use std::setw(4) for pretty-printing
    std::cout << "Log completed" << std::endl;
    std::cout << "Data written to: "<< filename << std::endl;
}

