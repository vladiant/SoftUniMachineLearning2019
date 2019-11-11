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
  constexpr auto file_name = "adult.data";

  // Default responses column = last
  // Variables types set according to OpenCV rules
  // Default delimiter = ','
  // Default missing data fill = '?'
  auto raw_data = cv::ml::TrainData::loadFromCSV(
      file_name,
      0,   // Header lines not to be read as data
      -1,  // Last columns is the target (Default)
      -1,  // Traget is one column (Default)
      "ord[0,2,10,11,12]cat[1,3,4-9,13,14]", ',', 'U');

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

  // Show target names
  // It seems that _all_ the categorical variables are indexed!
  std::vector<cv::String> target_names;
  raw_data->getNames(target_names);
  for (uint i = 0; i < target_names.size(); i++) {
    std::cout << "Target index: " << i << "  name: " << target_names[i]
              << std::endl;
  }

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

  auto gradient_boosted_trees = cv::ml::Boost::create();

  // If CVFolds > 1 then algorithms prunes the built decision tree using K-fold
  // cross-validation procedure where K is equal to CVFolds.
  // Values different than 0 and 1 cause crash!
  gradient_boosted_trees->setCVFolds(0);

  // This needs to be set properly
  // Default value is INT_MAX
  // This causes crash!
  gradient_boosted_trees->setMaxDepth(1);

  // Note: Bigger tree depth causes overfitting!

  //  cv::ml::Boost::DISCRETE, REAL, LOGIT, GENTLE
  gradient_boosted_trees->setBoostType(cv::ml::Boost::GENTLE);

  //  The number of weak classifiers. Default - 100
  gradient_boosted_trees->setWeakCount(200);

  // A threshold between 0 and 1 used to save computational time.
  // Default = 0.95
  // 0 -> Turns off functionality
  gradient_boosted_trees->setWeightTrimRate(0.95);

  // As Decision Tree
  //  gradient_boosted_trees->setMaxCategories(10);
  //  gradient_boosted_trees->setMinSampleCount(10);

  //  TODO: Ivestigate whether prior has to be applied for
  //  non-uniform distribution of labels to enforce stratification
  //  gradient_boosted_trees->setPriors(cv::Mat());

  //  gradient_boosted_trees->setRegressionAccuracy(0.01f);
  //  gradient_boosted_trees->setTruncatePrunedTree(true);
  //  gradient_boosted_trees->setUse1SERule(true);
  //  Value 'true' not implemented!
  //  gradient_boosted_trees->setUseSurrogates(false);

  std::cout << "Start training" << std::endl;

  // Train with the whole dataset
  gradient_boosted_trees->train(raw_data);

  // TODO: Investigate why this crashes
  //  gradient_boosted_trees->train(raw_data->getTrainSamples(),
  //                                cv::ml::SampleTypes::COL_SAMPLE,
  //                                raw_data->getTrainResponses());

  // For regression models the error is computed as RMS,
  // for classifiers - as a percent of missclassified samples (0%-100%).
  // Last argument - array to store the prediction results

  // False - tested on train data
  std::cout << "Missclassified train samples, %: "
            << gradient_boosted_trees->calcError(raw_data, false, cv::noArray())
            << std::endl;

  // True - tested on test data
  std::cout << "Missclassified test samples, %: "
            << gradient_boosted_trees->calcError(raw_data, true, cv::noArray())
            << std::endl;

  // TODO: Predicted values are float numbers. They may represent possibility
  // for a label. However it is unclear how the possibility can be calculated
  // for more than two labels. This needs to be investigated.

  // NOTE: In
  // https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_21-01.cpp
  // the target is the first column, so the first labels will be the target
  // ones?

  std::cout << "Done." << std::endl;
  return 0;
}
