#include <iostream>
#include <opencv/cv.hpp>
#include <opencv2/ml.hpp>

void PrintSetHistogram(const cv::Mat& set) {
  std::map<int, int> values_hist;
  for (int i = 0; i < set.rows; i++) {
    const int train_label = set.at<float>(i, 0);
    auto it = values_hist.find(train_label);
    if (values_hist.end() == it) {
      values_hist[train_label] = 1;
    } else {
      it->second++;
    }
  }

  for (const auto& hist_elem : values_hist) {
    std::cout << "Label: " << hist_elem.first
              << "  Count : " << hist_elem.second << std::endl;
  }
}

int main(int argc, char* argv[]) {
  constexpr auto file_name = "iris.data";

  // Default responses column = last
  // Variables types set according to OpenCV rules
  // Default delimiter = ','
  // Default missing data fill = '?'
  auto raw_data =
      cv::ml::TrainData::loadFromCSV(file_name,
                                     0  // Header lines not to be read as data
      );

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
  raw_data->setTrainTestSplitRatio(0.80, true);

  const int n_train_samples = raw_data->getNTrainSamples();
  const int n_test_samples = raw_data->getNTestSamples();
  std::cout << "Found " << n_train_samples << " Train Samples, and "
            << n_test_samples << " Test Samples" << std::endl;

  // Statistics of all data data
  std::cout << "All data data:" << std::endl;
  PrintSetHistogram(raw_data->getResponses());

  // Statistics of train data
  std::cout << "Train data:" << std::endl;
  PrintSetHistogram(raw_data->getTrainResponses());

  // Statistics of test data
  std::cout << "Test data:" << std::endl;
  PrintSetHistogram(raw_data->getTestResponses());

  std::cout << "Data loaded" << std::endl;

  auto svm_model = cv::ml::SVM::create();

  svm_model->setKernel(cv::ml::SVM::RBF);
  svm_model->setGamma(0.5);

  svm_model->setType(cv::ml::SVM::C_SVC);

  // Outlier penalization
  svm_model->setC(1.0);

  std::cout << "Start traning" << std::endl;

  // Train with the whole dataset
  svm_model->train(raw_data);

  // Bad argument (in the case of classification problem the responses must be
  // categorical; either specify varType when creating TrainData, or pass
  // integer responses) in train
  //  svm_model->train(raw_data->getTrainSamples(), 0,
  //  raw_data->getTrainResponses());

  // For regression models the error is computed as RMS,
  // for classifiers - as a percent of missclassified samples (0%-100%).
  // Last argument - array to store the prediction results

  // False - tested on train data
  std::cout << "Missclassified train samples, %: "
            << svm_model->calcError(raw_data, false, cv::noArray())
            << std::endl;

  for (int i = 0; i < raw_data->getTrainSamples().rows; i++) {
    const auto predicted =
        svm_model->predict(raw_data->getTrainSamples().row(i));
    const auto expected = raw_data->getTrainResponses().at<float>(i, 0);
    if (predicted != expected)
      std::cout << "Predicted: " << predicted << "  Expected: " << expected
                << std::endl;
  }

  // True - tested on test data
  std::cout << "Missclassified test samples, %: "
            << svm_model->calcError(raw_data, true, cv::noArray()) << std::endl;

  for (int i = 0; i < raw_data->getTestSamples().rows; i++) {
    const auto predicted =
        svm_model->predict(raw_data->getTestSamples().row(i));
    const auto expected = raw_data->getTestResponses().at<float>(i, 0);
    if (predicted != expected)
      std::cout << "Predicted: " << predicted << "  Expected: " << expected
                << std::endl;
  }

  std::cout << "Done." << std::endl;
  return 0;
}
