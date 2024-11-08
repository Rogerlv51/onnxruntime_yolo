// ONNXRUNTIME写代码大概分为三部分
// 1.初始化环境，会话等
// 2.会话中加载模型，得到模型的输入和输出节点
// 3.调用API得到模型的返回

#include "yolov10.h"

int YOLO_V10::init(DL_INIT_PARAM& iParams) {
	if (0 != _access(iParams.modelPath.c_str(), 0)) {
		std::cout << "Model path does not exist,  please check " << iParams.modelPath << std::endl;
		return 0;
	}
	else {
		std::cout << "读取Model path: " << iParams.modelPath << std::endl;
	}
	
	try {
		rectConfidenceThreshold = iParams.rectConfidenceThreshold;
		iouThreshold = iParams.iouThreshold;
		imgSize = iParams.imgSize;
		modelType = iParams.modelType;

		Ort::SessionOptions sessionOption;   // 配置会话属性
		if (iParams.cudaEnable)
		{
			cudaEnable = iParams.cudaEnable;
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = iParams.device_id;   // 设置设备ID
			sessionOption.AppendExecutionProvider_CUDA(cudaOption);
			std::cout << "正在使用GPU" << cudaOption.device_id << "进行推理" << std::endl;
		}
		else {
            std::cout << "正在使用CPU进行推理" << std::endl;
		}
		sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);  // 设置线程数
		sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);
		env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo_v10");
#ifdef _WIN32
		int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
		wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
		MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
		wide_cstr[ModelPathSize] = L'\0';
		const wchar_t* modelPath = wide_cstr;
#else
		const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32
		session = new Ort::Session(env, modelPath, sessionOption);
	}catch (const std::exception& e) {
		std::cout << "onnx文件异常" << std::endl;
	}

	return 0;
}

void YOLO_V10::preProcess(cv::Mat& img) {
	return;
}

void YOLO_V10::postProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg){
	return;
}

int YOLO_V10::inference(cv::Mat& img) {
	return 0;
}

YOLO_V10::YOLO_V10() {

}

YOLO_V10::~YOLO_V10() {
	if (session != nullptr) {
		delete session;
	}
};