#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_SCOPER_TIMER_HPP
#define POLIVEM_SCOPER_TIMER_HPP

#include <cmath>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <string>

namespace utils{
    class ScopeTimer{
    
    private:
        std::string name;
        std::chrono::high_resolution_clock::time_point start;

    public:
        ScopeTimer(const std::string& timer_name) 
            : name(timer_name), start(std::chrono::high_resolution_clock::now()) {}
        
        ~ScopeTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << name << " took: " << duration << " ms" << std::endl;
        }
    };
}

#endif