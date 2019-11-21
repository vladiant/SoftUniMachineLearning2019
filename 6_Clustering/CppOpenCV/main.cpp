#include <iostream>
#include <opencv/cv.hpp>
#include <opencv2/ml.hpp>

int main(int argc, char *argv[]) {
  constexpr uint64_t kSeed = 123456;

  constexpr int kNumSamples = 100;

  constexpr float kMeanX1 = 0.7;
  constexpr float kStdX1 = 0.15;

  constexpr float kMeanY1 = 0.7;
  constexpr float kStdY1 = 0.15;

  constexpr float kMeanX2 = 0.3;
  constexpr float kStdX2 = 0.15;

  constexpr float kMeanY2 = 0.3;
  constexpr float kStdY2 = 0.15;

  std::cout << "Start Clustring Demo" << std::endl;

  cv::Mat data_center_x_1(kNumSamples / 2, 1, CV_32F);
  cv::Mat data_center_x_2(kNumSamples / 2, 1, CV_32F);

  cv::Mat data_center_y_1(kNumSamples / 2, 1, CV_32F);
  cv::Mat data_center_y_2(kNumSamples / 2, 1, CV_32F);

  cv::RNG generator(kSeed);

  generator.fill(data_center_x_1, cv::RNG::NORMAL, kMeanX1, kStdX1);
  generator.fill(data_center_y_1, cv::RNG::NORMAL, kMeanY1, kStdY1);
  generator.fill(data_center_x_2, cv::RNG::NORMAL, kMeanX2, kStdX2);
  generator.fill(data_center_y_2, cv::RNG::NORMAL, kMeanY2, kStdY2);

  cv::Mat samples(kNumSamples, 2, CV_32F);
  cv::Mat responses(kNumSamples, 1, CV_32S);

  int i = 0;
  for (int j = 0; j < data_center_x_1.rows; i++, j++) {
    samples.at<float>(i, 0) = data_center_x_1.at<float>(j, 0);
    samples.at<float>(i, 1) = data_center_y_1.at<float>(j, 0);
    responses.at<int32_t>(i, 0) = 1;
  }

  for (int j = 0; j < data_center_x_2.rows; i++, j++) {
    samples.at<float>(i, 0) = data_center_x_2.at<float>(j, 0);
    samples.at<float>(i, 1) = data_center_y_1.at<float>(j, 0);
    responses.at<int32_t>(i, 0) = 0;
  }

  auto train_data =
      cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE, responses);

  std::cout << "Test data created" << std::endl;

  // Split train/test data; False - no shuffle
  train_data->setTrainTestSplitRatio(0.70, true);

  std::cout << "Total samples: " << train_data->getNSamples() << std::endl;

  const int n_train_samples = train_data->getNTrainSamples();
  const int n_test_samples = train_data->getNTestSamples();
  std::cout << "Split to " << n_train_samples << " Train Samples and "
            << n_test_samples << " Test Samples" << std::endl;

  auto knn = cv::ml::KNearest::create();

  // cv::ml::KNearest::BRUTE_FORCE, KDTREE
  knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);

  // Default number of neighbors to use in predict method
  knn->setDefaultK(4);

  // Parameter for KDTree implementation.
  // knn->setEmax(int);

  // Train with the whole dataset
  knn->train(train_data);

  // False - tested on train data
  std::cout << "Missclassified train samples, %: "
            << knn->calcError(train_data, false, cv::noArray()) << std::endl;

  // True - tested on test data
  std::cout << "Missclassified test samples, %: "
            << knn->calcError(train_data, true, cv::noArray()) << std::endl;

  return 0;
}
