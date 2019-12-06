# Rumale

![Rumale](https://dl.dropboxusercontent.com/s/joxruk2720ur66o/rumale_header_400.png)

[![Build Status](https://travis-ci.org/yoshoku/rumale.svg?branch=master)](https://travis-ci.org/yoshoku/rumale)
[![Coverage Status](https://coveralls.io/repos/github/yoshoku/rumale/badge.svg?branch=master)](https://coveralls.io/github/yoshoku/rumale?branch=master)
[![Gem Version](https://badge.fury.io/rb/rumale.svg)](https://badge.fury.io/rb/rumale)
[![BSD 2-Clause License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://github.com/yoshoku/rumale/blob/master/LICENSE.txt)
[![Documentation](http://img.shields.io/badge/docs-rdoc.info-blue.svg)](https://yoshoku.github.io/rumale/doc/)

Rumale (**Ru**by **ma**chine **le**arning) is a machine learning library in Ruby.
Rumale provides machine learning algorithms with interfaces similar to Scikit-Learn in Python.
Rumale supports Support Vector Machine,
Logistic Regression, Ridge, Lasso, Factorization Machine,
Multi-layer Perceptron,
Naive Bayes, Decision Tree, Gradient Tree Boosting, Random Forest,
K-Means, Gaussian Mixture Model, DBSCAN, Spectral Clustering,
Mutidimensional Scaling, t-SNE, Principal Component Analysis, Non-negative Matrix Factorization,
and many other algorithms.

This project was formerly known as "SVMKit".
If you are using SVMKit, please install Rumale and replace `SVMKit` constants with `Rumale`.

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'rumale'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install rumale

## Usage

### Example 1. XOR data
First, let's classify simple xor data.

```ruby
require 'rumale'

# Prepare XOR data.
features = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 1, 1, 0]

# Train classifier with nearest neighbor rule.
estimator = Rumale::NearestNeighbors::KNeighborsClassifier.new(n_neighbors: 1)
estimator.fit(x, y)

# Predict labels.
p y
p estimator.predict(x)
```

Execution of the above script result in the following.

```ruby
Numo::Int32#shape=[4]
[0, 1, 1, 0]
Numo::Int32#shape=[4]
[0, 1, 1, 0]
```

The basic usage of Rumale is to first train the model with the fit method
and then estimate with the predict method.
In addition, Rumale recommends using arrays such as feature vectors and labels with
[Numo::NArray](https://github.com/ruby-numo/numo-narray).

### Example 2. Pendigits dataset classification

Rumale provides function loading libsvm format dataset file.
We start by downloading the pendigits dataset from LIBSVM Data web site.

```bash
$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits
$ wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t
```

Training of the classifier with Linear SVM and RBF kernel feature map is the following code.

```ruby
require 'rumale'

# Load the training dataset.
samples, labels = Rumale::Dataset.load_libsvm_file('pendigits')

# Map training data to RBF kernel feature space.
transformer = Rumale::KernelApproximation::RBF.new(gamma: 0.0001, n_components: 1024, random_seed: 1)
transformed = transformer.fit_transform(samples)

# Train linear SVM classifier.
classifier = Rumale::LinearModel::SVC.new(reg_param: 0.0001, max_iter: 1000, batch_size: 50, random_seed: 1)
classifier.fit(transformed, labels)

# Save the model.
File.open('transformer.dat', 'wb') { |f| f.write(Marshal.dump(transformer)) }
File.open('classifier.dat', 'wb') { |f| f.write(Marshal.dump(classifier)) }
```

Classifying testing data with the trained classifier is the following code.

```ruby
require 'rumale'

# Load the testing dataset.
samples, labels = Rumale::Dataset.load_libsvm_file('pendigits.t')

# Load the model.
transformer = Marshal.load(File.binread('transformer.dat'))
classifier = Marshal.load(File.binread('classifier.dat'))

# Map testing data to RBF kernel feature space.
transformed = transformer.transform(samples)

# Classify the testing data and evaluate prediction results.
puts("Accuracy: %.1f%%" % (100.0 * classifier.score(transformed, labels)))

# Other evaluating approach
# results = classifier.predict(transformed)
# evaluator = Rumale::EvaluationMeasure::Accuracy.new
# puts("Accuracy: %.1f%%" % (100.0 * evaluator.score(results, labels)))
```

Execution of the above scripts result in the following.

```bash
$ ruby train.rb
$ ruby test.rb
Accuracy: 98.4%
```

### Example 3. Cross-validation

```ruby
require 'rumale'

# Load dataset.
samples, labels = Rumale::Dataset.load_libsvm_file('pendigits')

# Define the estimator to be evaluated.
lr = Rumale::LinearModel::LogisticRegression.new(reg_param: 0.0001, random_seed: 1)

# Define the evaluation measure, splitting strategy, and cross validation.
ev = Rumale::EvaluationMeasure::LogLoss.new
kf = Rumale::ModelSelection::StratifiedKFold.new(n_splits: 5, shuffle: true, random_seed: 1)
cv = Rumale::ModelSelection::CrossValidation.new(estimator: lr, splitter: kf, evaluator: ev)

# Perform 5-cross validation.
report = cv.perform(samples, labels)

# Output result.
mean_logloss = report[:test_score].inject(:+) / kf.n_splits
puts("5-CV mean log-loss: %.3f" % mean_logloss)
```

Execution of the above scripts result in the following.

```bash
$ ruby cross_validation.rb
5-CV mean log-loss: 0.476
```

### Example 4. Pipeline

```ruby
require 'rumale'

# Load dataset.
samples, labels = Rumale::Dataset.load_libsvm_file('pendigits')

# Construct pipeline with kernel approximation and SVC.
rbf = Rumale::KernelApproximation::RBF.new(gamma: 0.0001, n_components: 800, random_seed: 1)
svc = Rumale::LinearModel::SVC.new(reg_param: 0.0001, max_iter: 1000, random_seed: 1)
pipeline = Rumale::Pipeline::Pipeline.new(steps: { trns: rbf, clsf: svc })

# Define the splitting strategy and cross validation.
kf = Rumale::ModelSelection::StratifiedKFold.new(n_splits: 5, shuffle: true, random_seed: 1)
cv = Rumale::ModelSelection::CrossValidation.new(estimator: pipeline, splitter: kf)

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

## Speeding up

### Numo::Linalg
Loading the [Numo::Linalg](https://github.com/ruby-numo/numo-linalg) allows to perform matrix product of Numo::NArray using BLAS libraries.
For example, using the [OpenBLAS](https://github.com/xianyi/OpenBLAS) speeds up many estimators in Rumale.

Install OpenBLAS library.

Mac:

```bash
$ brew install openblas
```

Ubuntu:

```bash
$ sudo apt-get install gcc gfortran
$ wget https://github.com/xianyi/OpenBLAS/archive/v0.3.5.tar.gz
$ tar xzf v0.3.5.tar.gz
$ cd OpenBLAS-0.3.5
$ make USE_OPENMP=1
$ sudo make PREFIX=/usr/local install
```

Install Numo::Linalg gem.

```bash
$ gem install numo-linalg
```

In ruby script, you only need to require the autoloader module of Numo::Linalg.

```ruby
require 'numo/linalg/autoloader'
require 'rumale'
```

### Parallel
Several estimators in Rumale support parallel processing.
Parallel processing in Rumale is realized by [Parallel](https://github.com/grosser/parallel) gem,
so install and load it.

```bash
$ gem install parallel
```

```ruby
require 'parallel'
require 'rumale'
```

Estimators that support parallel processing have n_jobs parameter.
When -1 is given to n_jobs parameter, all processors are used.

```ruby
estimator = Rumale::Ensemble::RandomForestClassifier.new(n_jobs: -1, random_seed: 1)
```


## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/yoshoku/rumale.
This project is intended to be a safe, welcoming space for collaboration,
and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The gem is available as open source under the terms of the [BSD 2-clause License](https://opensource.org/licenses/BSD-2-Clause).

## Code of Conduct

Everyone interacting in the Rumale project’s codebases, issue trackers,
chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/yoshoku/Rumale/blob/master/CODE_OF_CONDUCT.md).
