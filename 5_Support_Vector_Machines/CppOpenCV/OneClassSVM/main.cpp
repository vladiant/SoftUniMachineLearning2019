#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

int main(int argc, char* argv[]) {
  constexpr auto file_name = "housing.data";

  // Default responses column = last
  // Variables types set according to OpenCV rules
  // Default delimiter = ','
  // Default missing data fill = '?'
  auto raw_data = cv::ml::TrainData::loadFromCSV(
      file_name,
      0,   // Header lines not to be read as data
      -1,  // Last columns is the target (Default)
      -1,  // Traget is one column (Default)
      "ord[0-2,4-7,9-13]cat[3, 8]", ',', 'U');

  const int n_samples = raw_data->getNSamples();
  if (n_samples == 0) {
    std::cerr << "Could not read file: " << file_name << std::endl;
    exit(-1);
  }

  std::cout << "Read " << n_samples << " samples from " << file_name
            << std::endl;

  // Split train/test data; False - no shuffle
  // When 'false' is selected, then the first data is selected
  // This may introduce bias in training due to uneven samples per label
  // distribution When 'true' is selected, the splitting is done by random Still
  // the samples distribution is not accounted Therefore stratification needs to
  // be introduced.
  raw_data->setTrainTestSplitRatio(0.70, true);

  const int n_train_samples = raw_data->getNTrainSamples();
  const int n_test_samples = raw_data->getNTestSamples();
  std::cout << "Found " << n_train_samples << " Train Samples, and "
            << n_test_samples << " Test Samples" << std::endl;

  std::cout << "Data loaded" << std::endl;

  auto svm_model = cv::ml::SVM::create();

  svm_model->setType(cv::ml::SVM::ONE_CLASS);
  svm_model->setKernel(cv::ml::SVM::POLY);

  svm_model->setDegree(3);
  svm_model->setGamma(1.0 / raw_data->getNVars());
  svm_model->setNu(0.01);
  svm_model->setCoef0(0.0);

  std::cout << "Start traning" << std::endl;

  // Train with the whole dataset
  //  svm_model->train(raw_data);
  svm_model->train(raw_data->getTrainSamples(), 0,
                   raw_data->getTrainResponses());

  // This algorithm is used to detect outliers
  // Predicted value 1 signifies inilier
  // Predicted value 0 signifies outlier

  std::cout << "Train samples ouliers" << std::endl;
  int train_outliers = 0;
  for (int i = 0; i < raw_data->getTrainSamples().rows; i++) {
    const int predicted =
        svm_model->predict(raw_data->getTrainSamples().row(i));
    if (predicted != 1) {
      train_outliers++;
    }
  }
  std::cout << "Train samples ouliers % "
            << 1.0 * train_outliers / raw_data->getNTrainSamples() << std::endl;

  std::cout << "Test samples ouliers" << std::endl;
  int test_outliers = 0;
  for (int i = 0; i < raw_data->getTestSamples().rows; i++) {
    const int predicted = svm_model->predict(raw_data->getTestSamples().row(i));
    if (predicted != 1) {
      test_outliers++;
    }
  }
  std::cout << "Test samples ouliers % "
            << 1.0 * test_outliers / raw_data->getNTestSamples() << std::endl;

  std::cout << "Done." << std::endl;
  return 0;
}
