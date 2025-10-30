#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_LOGGING_HPP
#define POLIVEM_LOGGING_HPP

#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <map>
#include <variant>
#include <string>

#include <nlohmann/json.hpp>

namespace utils {
    class logging{
    public:
        // generate a date string with the current date
        std::string generateDateString();

        // generate timestamp
        std::string generateTimestamp();

        // build log file
        void buildLogFile(std::map<std::string, std::string>& dataMap);
    };
}

#endif 