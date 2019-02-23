# SVMKit

[![Build Status](https://travis-ci.org/yoshoku/svmkit.svg?branch=master)](https://travis-ci.org/yoshoku/svmkit)
[![Coverage Status](https://coveralls.io/repos/github/yoshoku/svmkit/badge.svg?branch=master)](https://coveralls.io/github/yoshoku/svmkit?branch=master)
[![Gem Version](https://badge.fury.io/rb/svmkit.svg)](https://badge.fury.io/rb/svmkit)
[![BSD 2-Clause License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://github.com/yoshoku/svmkit/blob/master/LICENSE.txt)
[![Documentation](http://img.shields.io/badge/docs-rdoc.info-blue.svg)](https://www.rubydoc.info/gems/svmkit/)

SVMKit has been deprecated and has been renamed to [Rumale](https://github.com/yoshoku/rumale).
Initially, I started developing SVMKit as an experimental library aiming at implementing SVM in Ruby.
However, since I added many other machine learning algorithms to SVMKit, I decided to change the library name.
SVMKit will continue releasing for bugfix but will not add new features.

SVMKit is a machine learninig library in Ruby.
SVMKit provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
SVMKit supports Linear / Kernel Support Vector Machine,
Logistic Regression, Linear Regression, Ridge, Lasso, Factorization Machine,
Naive Bayes, Decision Tree, AdaBoost, Random Forest, K-nearest neighbor classifier,
K-Means, DBSCAN, Principal Component Analysis, and Non-negative Matrix Factorization.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'svmkit'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install svmkit

## Usage

### Example 1. Pendigits dataset classification

SVMKit provides function loading libsvm format dataset file.
We start by downloading the pendigits dataset from LIBSVM Data web site.

```bash
$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits
$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t
```

Training of the classifier with Linear SVM and RBF kernel feature map is the following code.

```ruby
require 'svmkit'

# Load the training dataset.
samples, labels = SVMKit::Dataset.load_libsvm_file('pendigits')

# Map training data to RBF kernel feature space.
transformer = SVMKit::KernelApproximation::RBF.new(gamma: 0.0001, n_components: 1024, random_seed: 1)
transformed = transformer.fit_transform(samples)

# Train linear SVM classifier.
classifier = SVMKit::LinearModel::SVC.new(reg_param: 0.0001, max_iter: 1000, batch_size: 50, random_seed: 1)
classifier.fit(transformed, labels)

# Save the model.
File.open('transformer.dat', 'wb') { |f| f.write(Marshal.dump(transformer)) }
File.open('classifier.dat', 'wb') { |f| f.write(Marshal.dump(classifier)) }
```

Classifying testing data with the trained classifier is the following code.

```ruby
require 'svmkit'

# Load the testing dataset.
samples, labels = SVMKit::Dataset.load_libsvm_file('pendigits.t')

# Load the model.
transformer = Marshal.load(File.binread('transformer.dat'))
classifier = Marshal.load(File.binread('classifier.dat'))

# Map testing data to RBF kernel feature space.
transformed = transformer.transform(samples)

# Classify the testing data and evaluate prediction results.
puts("Accuracy: %.1f%%" % (100.0 * classifier.score(transformed, labels)))

# Other evaluating approach
# results = classifier.predict(transformed)
# evaluator = SVMKit::EvaluationMeasure::Accuracy.new
# puts("Accuracy: %.1f%%" % (100.0 * evaluator.score(results, labels)))
```

Execution of the above scripts result in the following.

```bash
$ ruby train.rb
$ ruby test.rb
Accuracy: 98.4%
```

### Example 2. Cross-validation

```ruby
require 'svmkit'

# Load dataset.
samples, labels = SVMKit::Dataset.load_libsvm_file('pendigits')

# Define the estimator to be evaluated.
lr = SVMKit::LinearModel::LogisticRegression.new(reg_param: 0.0001, random_seed: 1)

# Define the evaluation measure, splitting strategy, and cross validation.
ev = SVMKit::EvaluationMeasure::LogLoss.new
kf = SVMKit::ModelSelection::StratifiedKFold.new(n_splits: 5, shuffle: true, random_seed: 1)
cv = SVMKit::ModelSelection::CrossValidation.new(estimator: lr, splitter: kf, evaluator: ev)

# Perform 5-cross validation.
report = cv.perform(samples, labels)

# Output result.
mean_logloss = report[:test_score].inject(:+) / kf.n_splits
puts("5-CV mean log-loss: %.3f" % mean_logloss)
```

### Example 3. Pipeline

```ruby
require 'svmkit'

# Load dataset.
samples, labels = SVMKit::Dataset.load_libsvm_file('pendigits')

# Construct pipeline with kernel approximation and SVC.
rbf = SVMKit::KernelApproximation::RBF.new(gamma: 0.0001, n_components: 800, random_seed: 1)
svc = SVMKit::LinearModel::SVC.new(reg_param: 0.0001, max_iter: 1000, random_seed: 1)
pipeline = SVMKit::Pipeline::Pipeline.new(steps: { trns: rbf, clsf: svc })

# Define the splitting strategy and cross validation.
kf = SVMKit::ModelSelection::StratifiedKFold.new(n_splits: 5, shuffle: true, random_seed: 1)
cv = SVMKit::ModelSelection::CrossValidation.new(estimator: pipeline, splitter: kf)

# Perform 5-cross validation.
report = cv.perform(samples, labels)

# Output result.
mean_accuracy = report[:test_score].inject(:+) / kf.n_splits
puts("5-CV mean accuracy: %.1f %%" % (mean_accuracy * 100.0))
```

Execution of the above scripts result in the following.

```bash
$ ruby pipeline.rb
5-CV mean accuracy: 99.2 %
```

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yoshoku/svmkit.
This project is intended to be a safe, welcoming space for collaboration,
and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The gem is available as open source under the terms of the [BSD 2-clause License](https://opensource.org/licenses/BSD-2-Clause).

## Code of Conduct

Everyone interacting in the SVMKit project’s codebases, issue trackers,
chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/yoshoku/svmkit/blob/master/CODE_OF_CONDUCT.md).
