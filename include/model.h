#pragma once
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
enum MODEL_TYPE  // 任务类型
{
    //FLOAT32 MODEL
    YOLO_DETECT_V10 = 1,
    YOLO_SEGMENT_V10 = 2,
    YOLO_CLS = 3,

    //FLOAT16 MODEL
    YOLO_DETECT_V10_HALF = 4,
    YOLO_SEGMENT_V10_HALF = 5,
    YOLO_CLS_HALF = 6
};


typedef struct _DL_INIT_PARAM   // 初始化参数
{
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_DETECT_V10;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.5;
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
    int device_id = 0;
} DL_INIT_PARAM;


typedef struct _DL_RESULT   // 保存模型输出结果
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Mat> Mask;
} DL_RESULT;

// 虚基类，后续不同版本yolo可以继承重写函数即可
class Model {
public:
    virtual ~Model(){};
    virtual int inference(cv::Mat& img) = 0;
    virtual int init(DL_INIT_PARAM& iParams) = 0;
    
    virtual void preProcess(cv::Mat& img) = 0;
    virtual void postProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg) = 0;
};