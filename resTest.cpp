#include <vector>
#include <string.h>

#include <caffe/caffe.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace caffe;
using namespace std;

int main(){ 
  
  // Load the network
  string proto = "scripts/deploy_resnet50by2_pool.prototxt";
  Phase phase = TEST;
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);
  boost::shared_ptr< Net<float> > net(new caffe::Net<float>(proto, phase));
  string model = "model/train_iter_40000.caffemodel";
  net->CopyTrainedLayersFrom(model);
  
  CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly one output.";
  
  // Convert image to input blob. Blob: num, channel, height, width
  Blob<float>* input_blob = net->input_blobs()[0];  
  int input_channel = input_blob->channels();
  int input_height = input_blob->height();
  int input_width = input_blob->width();
  cout << "The size of the network is " << input_width << "*" << input_height << endl;
  
  vector<cv::Mat> input_channels;  //The value of input image's each channel  
  float* input_data = input_blob->mutable_cpu_data();
  for(int i=0;i<input_channel;i++){
    cv::Mat channel(input_height, input_width, CV_32FC1, input_data);
    input_channels.push_back(channel);
    input_data += input_width*input_height;
  }
  
  string imgPath = "images/rgb2.png";
  cv::Mat image = cv::imread(imgPath);
  
  cv::Size input_size = cv::Size(input_width, input_height);
  cv::Mat image_resized;  
  cv::resize(image, image_resized, input_size);
  
  cv::imshow("image", image_resized);
  cv::waitKey(0);
  
  
  cv::Mat image_float;
  image_resized.convertTo(image_float, CV_32FC3);
  cv::Mat image_normalized;
  cv::Mat mean(input_height, input_width, CV_32FC3, cv::Scalar(104,117,123));
  cv::subtract(image_float, mean, image_normalized);
  cv::split(image_normalized, input_channels);
  
  net->Forward();
  
  // Convert output blob to image
  Blob<float>* output_blob = net->output_blobs()[0];
  int output_height = output_blob->height();
  int output_width = output_blob->width();
  float *output_data = output_blob->mutable_cpu_data();
  cv::Mat depth_resized(output_height, output_width, CV_32FC1, output_data);
  
  cv::Mat depth;
  cv::resize(depth_resized, depth, cv::Size(image.cols, image.rows) );
  
  cv::Mat depth_uint8;
  depth.convertTo(depth_uint8, CV_8UC1);
  cv::imwrite("./images/output2.png", depth_uint8);
  
  cv::imshow("depth", depth_uint8);
  cv::waitKey(0);
  
  
  // Generate 3D image with rgb image and estimated depth
  
  //cv::Mat image = cv::imread("images/000009_10_image.png");
  //cv::Mat depth = cv::imread("images/000009_10_depth.png", -1);
  /*
  typedef pcl::PointXYZRGBA PointT;
  typedef pcl::PointCloud<PointT> PointCloud;
  
  const double camera_factor = 1000.0*608.0/image.cols;
  const double camera_fx = 984.24;
  const double camera_cx = 690.00;
  const double camera_fy = 980.81;
  const double camera_cy = 233.20;
  /*
  const double camera_factor = 1000.0;
  const double camera_fx = 518.0;
  const double camera_cx = 325.5;
  const double camera_fy = 519.0;
  const double camera_cy = 253.5;
  
  PointCloud::Ptr cloud( new PointCloud);
  for(int m=0;m<depth.rows;m++){
    for(int n=0;n<depth.cols;n++){
      float d = depth.ptr<float>(m)[n];
      
      PointT p;
      p.z = d / camera_factor;
      p.x = (n - camera_cx)*p.z / camera_fx;
      p.y = (m - camera_cy)*p.z / camera_fy;      
      p.b = image.ptr<uchar>(m)[n*3];
      p.g = image.ptr<uchar>(m)[n*3+1];
      p.r = image.ptr<uchar>(m)[n*3+2];
      
      cloud->points.push_back( p );
    }
  }
  
  cloud->height = 1;
  cloud->width = cloud->points.size();
  cout << "Point cloud size = " << cloud->points.size() << endl;
  
  cloud->is_dense = false;
  pcl::io::savePCDFile("./pointcloud.pcd", *cloud);
  cloud->points.clear();
  cout << "Point cloud saved." << endl;
  */
  return 0;
}