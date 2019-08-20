#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <winsock2.h>
#include <conio.h> 
#include <cmath>
#include <setjmp.h>

#include "include/json/json.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "utils.h"
// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::uint8;

using namespace std;

/*********
 * 
 * 
 * 
 * Json Method (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
 * 
 * 
 **********/
Json::Value ToJsonValue(const string& json_str) {
  Json::Value root;
  Json::Reader reader;
  reader.parse(json_str, root);
  return root;
}

void PrintJsonData(size_t i, 
                  float score, 
                  int classes,
                  float left_top_x,
                  float left_top_y,
                  float right_bottom_x,
                  float right_bottom_y) {
    Json::Value event;   
    Json::Value top_left(Json::arrayValue);
    Json::Value bottom_right(Json::arrayValue);

    top_left.append(Json::Value((int)(left_top_x * 1280)));
    top_left.append(Json::Value((int)(left_top_y * 720)));
    bottom_right.append(Json::Value((int)(right_bottom_x * 1280)));
    bottom_right.append(Json::Value((int)(right_bottom_y * 720)));

    event[std::to_string(i)]["score"]=Json::Value(score);
    event[std::to_string(i)]["class"]=Json::Value(classes);
    event[std::to_string(i)]["top_left"]=top_left;
    event[std::to_string(i)]["bottom_right"]=bottom_right;

    std::cout << event << std::endl;
}
/*********
 * 
 * 
 * 
 * main function
 * 
 * 
 * 
 *********/

int main(int argc, char* argv[]) {
  string image = "test_img/bluebox1057.jpg";
  string graph = "8_2_18_Inception.pb";
  string label = "label_map.pbtxt";

  int32 input_width = 299;
  int32 input_height = 299;
  float input_mean = 0;
  float input_std = 255;
  string input_layer = "image_tensor:0";
  vector<string> output_layer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

  bool self_test = false;
  string root_dir = "tensorflow/cc/workspace/Data/";

  // TO DO : ADD FLAG PARSE
    std::vector<Flag> flag_list = {
    Flag("image", &image, "image to be precessed"),
    Flag("graph", &graph, "graph to be executed"),
    Flag("label", &label, "name of label_map"),
    Flag("input_width", &input_width, "resize image to this width in pixels"),
    Flag("input_height", &input_height, "resize image to this height in pixels"),
    Flag("input_mean", &input_mean, "scale pixel value to this mean"),
    Flag("input_std", &input_std, "scale pixel values to this std deviation"),
    Flag("root_dir", &root_dir, "interpret image and graph file names relative to this directory"),
    // Flag("input_layer", &input_layer, "name of input layer"),
    // Flag("output_layer", &output_layer, "name of output layer"),
    // Flag("self_test", &self_test, "run a self test"),
  };
  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  LOG(ERROR) << "graph_path:" << graph_path;
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << "LoadGraph ERROR!!!!"<< load_graph_status;
    return -1;
  }

  
   
  std:: clock_t program_start = std::clock();
  double program_duration;
  while (true) {
    printf("waiting for signal!!!\n");
    char ch;
    ch = getch();

    if (ch == 'q') 
    {
      printf("process terminated\n");
      break;
    } 
    else if (ch == 'p') 
    {
      program_duration = (std::clock() - program_start) / (double) CLOCKS_PER_SEC;
      std::cout<<"time interval from program start until pressing p: "<<program_duration<<std::endl;
      
      // initialize a timer
      std:: clock_t process_start = std::clock();
      double process_duration;

      // image processing and print flatten output layer in Json
    std::vector<Tensor> resized_tensors;
    string curr_image_path = tensorflow::io::JoinPath(root_dir, image);
    Status read_tensor_status =
      ReadTensorFromImageFile(curr_image_path, input_height, input_width, input_mean,
                              input_std, &resized_tensors);
    if (!read_tensor_status.ok()) {
      LOG(ERROR) << read_tensor_status;
      return -1;
    }
    const Tensor& resized_tensor = resized_tensors[0];

    LOG(ERROR) <<"image shape:" << resized_tensor.shape().DebugString()<< ",len:" << resized_tensors.size() << ",tensor type:"<< resized_tensor.dtype();
    // << ",data:" << resized_tensor.flat<tensorflow::uint8>();
    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_layer, resized_tensor}},
                                     output_layer, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }
    int image_width = resized_tensor.dims();
    int image_height = 0;
    //int image_height = resized_tensor.shape()[1];

    //LOG(ERROR) << "size:" << outputs.size() << ",image_width:" << image_width << ",image_height:" << image_height << endl;

    //tensorflow::TTypes<float>::Flat iNum = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float,3>();


    LOG(ERROR) << "num_detections:" << num_detections(0) << "," << outputs[0].shape().DebugString();

    for(size_t i = 0; i < num_detections(0) && i < 20;++i)
    {
      if(scores(i) > 0.45)
      {
        LOG(ERROR) << i << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << "," << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);
        PrintJsonData(i, scores(i), classes(i), boxes(0,i,1), boxes(0,i,0), boxes(0,i,3), boxes(0,i,2));
      }
    }

      process_duration = (std::clock() - process_start) / (double) CLOCKS_PER_SEC;
      std::cout<<"runtime of current image processing"<<process_duration<<std::endl;
    }
  }
  return 0;
}

/*********
int main(int argc, char* argv[]) {
  // set default input path
  // if use something other than them, specify path by using Flag
  string image_root = "tensorflow/cc/workspace/Data/test_img";
  string graph = "tensorflow/cc/workspace/Data/8_2_18_Inception.pb";
  string label = "tensorflow/cc/workspace/Data/label_map.pbtxt";

  int32 input_width = 299;
  int32 input_height = 299;
  float input_mean = 0;
  float input_std = 255;
  string input_layer = "image_tensor:0";
  vector<string> output_layer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

  bool self_test = false;
  string root_dir = "";

  // TO DO : ADD FLAG PARSE

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  LOG(ERROR) << "graph_path:" << graph_path;
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << "LoadGraph ERROR!!!!"<< load_graph_status;
    return -1;
  }

  // retrieve all images in a directory
  vector<string> images;
  string image_path = tensorflow::io::JoinPath(root_dir, image_root);
  images.push_back("tensorflow/cc/workspace/Data/test_img/bluebox1004.jpg");
  images.push_back("tensorflow/cc/workspace/Data/test_img/bluebox1007.jpg");
  images.push_back("tensorflow/cc/workspace/Data/test_img/bluebox1014.jpg");
  images.push_back("tensorflow/cc/workspace/Data/test_img/bluebox1057.jpg");
  images.push_back("tensorflow/cc/workspace/Data/test_img/bluebox1076.jpg");

  // recursively fetch image from target folder
  int i;
  for (i=0; i < images.size(); i++) {
    // set a timer to show processing interval
    std::clock_t start = std::clock();
    double duration;

    std::vector<Tensor> resized_tensors;
    string curr_image_path = tensorflow::io::JoinPath(root_dir, images[i]);
    Status read_tensor_status =
      ReadTensorFromImageFile(curr_image_path, input_height, input_width, input_mean,
                              input_std, &resized_tensors);
    if (!read_tensor_status.ok()) {
      LOG(ERROR) << read_tensor_status;
      return -1;
    }
    const Tensor& resized_tensor = resized_tensors[0];

    LOG(ERROR) <<"image shape:" << resized_tensor.shape().DebugString()<< ",len:" << resized_tensors.size() << ",tensor type:"<< resized_tensor.dtype();
    // << ",data:" << resized_tensor.flat<tensorflow::uint8>();
    // Actually run the image through the model.
    std::vector<Tensor> outputs;
    Status run_status = session->Run({{input_layer, resized_tensor}},
                                     output_layer, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }
    int image_width = resized_tensor.dims();
    int image_height = 0;
    //int image_height = resized_tensor.shape()[1];

    //LOG(ERROR) << "size:" << outputs.size() << ",image_width:" << image_width << ",image_height:" << image_height << endl;

    //tensorflow::TTypes<float>::Flat iNum = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float,3>();


    LOG(ERROR) << "num_detections:" << num_detections(0) << "," << outputs[0].shape().DebugString();

    for(size_t i = 0; i < num_detections(0) && i < 20;++i)
    {
      if(scores(i) > 0.45)
      {
        LOG(ERROR) << i << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << "," << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);
        PrintJsonData(i, scores(i), classes(i), boxes(0,i,1), boxes(0,i,0), boxes(0,i,3), boxes(0,i,2));
      }
    }
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"time interval of image"<<i<<": "<<duration<<std::endl;  
  }

  // char keyBreak;
  // std::cout<<"start processing, press ese to exit"<<std::endl;
  // while(true) {
  //   keyBreak = getch();
  //   if (keyBreak == 27)
  //     break;
  //   }

  // std::cout<<"exited"<<std::endl;





  return 0;

}
**********/








