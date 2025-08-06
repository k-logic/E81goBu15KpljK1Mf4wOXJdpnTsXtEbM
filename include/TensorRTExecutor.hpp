#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <cassert>

#include "IModelExecutor.hpp"

using namespace nvinfer1; 

// TensorRT 実装
class TensorRTExecutor : public IModelExecutor {
public:
    TensorRTExecutor(cudaStream_t stream)
        : stream_(stream) {}

    ~TensorRTExecutor() {
        if (inputDev_) cudaFree(inputDev_);
        if (outputDev_) cudaFree(outputDev_);
    }

    void load(const std::string& enginePath) override {
        std::ifstream file(enginePath, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open engine: " << enginePath << std::endl;
            std::exit(1);
        }
        std::vector<char> engineData((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        
        Logger logger;
        runtime_.reset(createInferRuntime(logger));
        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
        context_.reset(engine_->createExecutionContext());

        inputName_ = engine_->getIOTensorName(0);
        outputName_ = engine_->getIOTensorName(1);
        inputSize_ = getTensorSize(engine_->getTensorShape(inputName_.c_str()));
        outputSize_ = getTensorSize(engine_->getTensorShape(outputName_.c_str()));

        if (cudaMalloc(&inputDev_, inputSize_ * sizeof(float)) != cudaSuccess) {
            std::cerr << "Failed to allocate inputDev_\n";
            std::exit(1);
        }
        if (cudaMalloc(&outputDev_, outputSize_ * sizeof(float)) != cudaSuccess) {
            std::cerr << "Failed to allocate outputDev_\n";
            std::exit(1);
        }
    }

    void run(const std::vector<float>& input, std::vector<float>& output) override {
        assert(input.size() == inputSize_);
        output.resize(outputSize_);

        cudaMemcpyAsync(inputDev_, input.data(), inputSize_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
        context_->setTensorAddress(inputName_.c_str(), inputDev_);
        context_->setTensorAddress(outputName_.c_str(), outputDev_);

        if (!context_->enqueueV3(stream_)) {
            std::cerr << "enqueueV3 failed.\n";
        }

        cudaMemcpyAsync(output.data(), outputDev_, outputSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    size_t getOutputSize() const { return outputSize_; }

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cout << "[TensorRT] " << msg << std::endl;
        }
    };

    std::unique_ptr<IRuntime> runtime_;
    std::unique_ptr<ICudaEngine> engine_;
    std::unique_ptr<IExecutionContext> context_;
    cudaStream_t stream_;

    std::string inputName_;
    std::string outputName_;
    void* inputDev_ = nullptr;
    void* outputDev_ = nullptr;
    size_t inputSize_ = 0;
    size_t outputSize_ = 0;

    size_t getTensorSize(const Dims& dims) {
        size_t size = 1;
        for (int i = 0; i < dims.nbDims; ++i)
            size *= dims.d[i];
        return size;
    }
};