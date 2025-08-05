#pragma once
#include <string>
#include <vector>

class IModelExecutor {
public:
    virtual ~IModelExecutor() = default;
    virtual void load(const std::string& enginePath) = 0;
    virtual void run(const std::vector<float>& input, std::vector<float>& output) = 0;
};