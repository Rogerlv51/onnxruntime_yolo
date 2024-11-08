#pragma once
#include "model.h"

#define    RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <cstdio>

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif


class YOLO_V10:public Model
{
public:
    YOLO_V10();

    ~YOLO_V10();

public:
    // 初始化
    int init(DL_INIT_PARAM& iParams);

    int inference(cv::Mat& img);

    void preProcess(cv::Mat& img);

    char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);

    char* WarmUpSession();

    template<typename N>
    char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);

    void postProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);

    std::vector<std::string> classes{};

private:
    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales;//letterbox scale
};