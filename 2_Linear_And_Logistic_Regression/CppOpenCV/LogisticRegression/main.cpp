#include <iostream>
#include <opencv/cv.hpp>
#include <opencv2/ml.hpp>

int main(int argc, char *argv[]) {
  auto raw_data = cv::ml::TrainData::loadFromCSV("iris.data", 0);

  cv::Mat attributes_data = raw_data->getSamples();
  cv::Mat labels_data = raw_data->getResponses();

  std::cout << "Data loaded" << std::endl;

  // [1; 2; 3] -> values for labels
  //    std::cout << raw_data->getClassLabels() << std::endl;

  //    Iris-setosa
  //    Iris-versicolor
  //    Iris-virginica
  //    std::vector<cv::String> names;
  //    raw_data->getNames(names);
  //    for(const auto& name: names) {
  //        std::cout << name << std::endl;
  //    }

  auto logistic_regression = cv::ml::LogisticRegression::create();

  logistic_regression->setLearningRate(0.1);
  logistic_regression->setIterations(100000);

  // Other values: cv::ml::LogisticRegression::REG_L1, REG_DISABLE , REG_L2
  logistic_regression->setRegularization(cv::ml::LogisticRegression::REG_L2);

  // Other values: cv::ml::LogisticRegression::MINI_BATCH , BATCH
  logistic_regression->setTrainMethod(cv::ml::LogisticRegression::BATCH);

  // MINI_BATCH only, positive integer
  // Does not work with this dataset
  // logistic_regression->setMiniBatchSize(3);

  std::cout << "Start traning" << std::endl;

  // Train with the whole dataset
  // logistic_regression->train(raw_data);
  logistic_regression->train(attributes_data, 0, labels_data);

  // For regression models the error is computed as RMS,
  // for classifiers - as a percent of missclassified samples (0%-100%).
  std::cout << "Missclassified samples, %: " << logistic_regression->calcError(raw_data, true, labels_data)
            << std::endl;

  logistic_regression->save("my.xml");
  // Load:
  // auto loaded_logistic_regression = cv::Algorithm::load<cv::ml::LogisticRegression>("my.xml");

  for (int i = 0; i < labels_data.rows; i++) {
    const auto predicted = logistic_regression->predict(attributes_data.row(i));
    const auto expected = labels_data.at<float>(i, 0);
    std::cout << "Predicted: " << predicted << "  Expected: " << expected << std::endl;
  }

  std::cout << "Done." << std::endl;
  return 0;
}
