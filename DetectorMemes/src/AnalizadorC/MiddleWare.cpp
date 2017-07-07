#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>


#include <boost/filesystem.hpp>


struct Context
{
  cv::Mat vocabulary;
  cv::FlannBasedMatcher flann;
  std::map<int, std::string> classes;
  cv::Ptr<cv::ml::ANN_MLP> mlp;
};



cv::Mat getDescriptors(const cv::Mat& img)
{
  cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
  return descriptors;
}

/**
 * Get a histogram of visual words for an image
 */
cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& img,
  int vocabularySize)
{
  cv::Mat descriptors = getDescriptors(img);
  cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
  std::vector<cv::DMatch> matches;
  flann.match(descriptors, matches);
  for (size_t j = 0; j < matches.size(); j++)
  {
      int visualWord = matches[j].trainIdx;
      outputArray.at<float>(visualWord)++;
  }
  return outputArray;
}

/**
 * Receives a column matrix contained the probabilities associated to 
 * each class and returns the id of column which contains the highest
 * probability
 */
int getPredictedClass(const cv::Mat& predictions)
{
  float maxPrediction = predictions.at<float>(0);
  float maxPredictionIndex = 0;
  const float* ptrPredictions = predictions.ptr<float>(0);
  for (int i = 0; i < predictions.cols; i++)
  {
      float prediction = *ptrPredictions++;
      if (prediction > maxPrediction)
      {
          maxPrediction = prediction;
          maxPredictionIndex = i;
      }
  }
  return maxPredictionIndex;
}

/**
 * Get the predicted class for a sample
 */
int getClass(const cv::Mat& bowFeatures, cv::Ptr<cv::ml::ANN_MLP> mlp)
{
  cv::Mat output;
  mlp->predict(bowFeatures, output);
  return getPredictedClass(output);
}




int main(int argc, char** argv){

std::string neuralNetworkInputFilename(argv[1]);
std::string vocabularyInputFilename(argv[2]);
std::string classesInputFilename(argv[3]);
std::string direccionMuestra(argv[4]);

//std::cout << "Training FLANN..." << std::endl;
cv::Ptr<cv::ml::ANN_MLP> mlp = cv::Algorithm::load<cv::ml::ANN_MLP>(neuralNetworkInputFilename);
//std::cout << "Training FLANN   2..." << std::endl;  
 cv::Mat vocabulary;
  cv::FileStorage fs(vocabularyInputFilename, cv::FileStorage::READ);
  fs["vocabulary"] >> vocabulary;
  fs.release();
//std::cout << "Training FLANN..." << std::endl;
///////Leyendo las classes
std::map<int,std::string> classes;
 std::ifstream classesInput(classesInputFilename.c_str());
  std::string line;
  while (std::getline(classesInput, line))
  {
      std::stringstream ss;
      ss << line;
      int index;
      std::string classname;
      ss >> index;
      ss >> classname;
      classes[index] = classname;
  }
//std::cout << "Training FLANN..." << std::endl;
 int start = cv::getTickCount();
  cv::FlannBasedMatcher flann;
  flann.add(vocabulary);
  flann.train();
  //std::cout << "Time elapsed in seconds: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() << std::endl;

///////////////////////////////
//std::cout<<"Antes"<<std::endl;
//std::cout<<direccionMuestra<<std::endl;
cv::Mat img = cv::imread(direccionMuestra,0);
Context* context = new Context;
context->flann = flann;
context->vocabulary=vocabulary;
context->classes = classes;
context->mlp=mlp;
//std::cout<<"Antes ++++++"<<std::endl;
 cv::Mat bowFeatures = getBOWFeatures(context->flann, img, context->vocabulary.rows);
//std::cout<<"Antes ++++++"<<std::endl;
              cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
//std::cout<<"Antes ++++++"<<std::endl;
              int predictedClass = getClass(bowFeatures, context->mlp);
//std::cout<<"Antes 123"<<std::endl;
              std::string result = context->classes[predictedClass];

std::cout <<result<<std::endl;
  
}
