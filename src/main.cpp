#include "yolov10.h"
#include <iostream>
using namespace std;

int main(){
    DL_INIT_PARAM dl_init_param;
    dl_init_param.modelPath = "yolov10s.onnx";
    //dl_init_param.cudaEnable = true;
    std::string img_path = "test.jpg";
    YOLO_V10 yolo;
    yolo.init(dl_init_param);
    return 0;
}