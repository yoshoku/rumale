require 'svmkit/base/splitter'

module SVMKit
  # This module consists of the classes for model validation techniques.
  module ModelSelection
    # CrossValidation is a class that performs (stratified) K-fold cross-validation.
    #
    # @example
    #   svc = SVMKit::LinearModel::SVC.new
    #   kf = SVMKit::ModelSelection::StratifiedKFold.new(n_splits: 5)
    #   cv = SVMKit::ModelSelection::CrossValidation.new(estimator: svc, splitter: kf)
    #   report = cv.perform(samples, lables)
    #   mean_test_score = report[:test_score].inject(:+) / kf.n_splits
    #
    class CrossValidation

      # Return the
      # @return [Classifier]
      attr_reader :estimator

      # Return the
      # @return [Splitter]
      attr_reader :splitter

      # Return the
      # @return [Hash]
      attr_reader :report

      # Return the
      # @return [Boolean]
      attr_reader :return_train_score

      # Create a new k-fold cross validator.
      #
      # @param estimator [Classifier] The classifier
      # @param splitter [Splitter] The number of folds.
      # @param return_train_score [Boolean] The seed value using to initialize the random generator.
      def initialize(estimator: nil, splitter: nil, return_train_score: false)
        @estimator = estimator
        @splitter = splitter
        @return_train_score = return_train_score
        @report = nil
      end

      # ABC
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The dataset to be used to generate data indices for K-fold cross validation.
      # @param y [Numo::Int32] (shape: [n_samples])
      #   The labels to be used to generate data indices for stratified K-fold cross validation.
      # @return [Array] The set of data indices for constructing the training and testing dataset in each fold.
      def perform(x, y)
        #
        @report = {test_score: [], train_score: nil, fit_time: []}
        @report[:train_score] = [] if @return_train_score
        #
        @splitter.split(x, y).each do |train_ids, test_ids|
          #
          feature_ids = !kernel_machine? || train_ids
          train_x = x[train_ids, feature_ids]
          train_y = y[train_ids]
          test_x = x[test_ids, feature_ids]
          test_y = y[test_ids]
          #
          start_time = Time.now.to_i
          @estimator.fit(train_x, train_y)
          #
          @report[:fit_time].push(Time.now.to_i - start_time)
          @report[:test_score].push(@estimator.score(test_x, test_y))
          @report[:train_score].push(@estimator.score(train_x, train_y)) if @return_train_score
        end
        @report
      end

      private

      def kernel_machine?
        @estimator.class.to_s.include?('KernelMachine')
      end
    end
  end
end
