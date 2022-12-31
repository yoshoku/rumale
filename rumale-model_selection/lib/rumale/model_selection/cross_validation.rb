# frozen_string_literal: true

require 'rumale/evaluation_measure/log_loss'

module Rumale
  # This module consists of the classes for model validation techniques.
  module ModelSelection
    # CrossValidation is a class that evaluates a given classifier with cross-validation method.
    #
    # @example
    #   require 'rumale/linear_model'
    #   require 'rumale/model_selection/stratified_k_fold'
    #   require 'rumale/model_selection/cross_validation'
    #
    #   svc = Rumale::LinearModel::SVC.new
    #   kf = Rumale::ModelSelection::StratifiedKFold.new(n_splits: 5)
    #   cv = Rumale::ModelSelection::CrossValidation.new(estimator: svc, splitter: kf)
    #   report = cv.perform(samples, labels)
    #   mean_test_score = report[:test_score].inject(:+) / kf.n_splits
    #
    class CrossValidation
      # Return the classifier of which performance is evaluated.
      # @return [Classifier]
      attr_reader :estimator

      # Return the splitter that divides dataset.
      # @return [Splitter]
      attr_reader :splitter

      # Return the evaluator that calculates score.
      # @return [Evaluator]
      attr_reader :evaluator

      # Return the flag indicating whether to caculate the score of training dataset.
      # @return [Boolean]
      attr_reader :return_train_score

      # Create a new evaluator with cross-validation method.
      #
      # @param estimator [Classifier] The classifier of which performance is evaluated.
      # @param splitter [Splitter] The splitter that divides dataset to training and testing dataset.
      # @param evaluator [Evaluator] The evaluator that calculates score of estimator results.
      # @param return_train_score [Boolean] The flag indicating whether to calculate the score of training dataset.
      def initialize(estimator: nil, splitter: nil, evaluator: nil, return_train_score: false)
        @estimator = estimator
        @splitter = splitter
        @evaluator = evaluator
        @return_train_score = return_train_score
      end

      # Perform the evalution of given classifier with cross-validation method.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features])
      #   The dataset to be used to evaluate the estimator.
      # @param y [Numo::Int32 / Numo::DFloat] (shape: [n_samples] / [n_samples, n_outputs])
      #   The labels to be used to evaluate the classifier / The target values to be used to evaluate the regressor.
      # @return [Hash] The report summarizing the results of cross-validation.
      #   * :fit_time (Array<Float>) The calculation times of fitting the estimator for each split.
      #   * :test_score (Array<Float>) The scores of testing dataset for each split.
      #   * :train_score (Array<Float>) The scores of training dataset for each split. This option is nil if
      #     the return_train_score is false.
      def perform(x, y)
        # Initialize the report of cross validation.
        report = { test_score: [], train_score: nil, fit_time: [] }
        report[:train_score] = [] if @return_train_score
        # Evaluate the estimator on each split.
        @splitter.split(x, y).each do |train_ids, test_ids|
          # Split dataset into training and testing dataset.
          feature_ids = !kernel_machine? || train_ids
          train_x = x[train_ids, feature_ids]
          train_y = y.shape[1].nil? ? y[train_ids] : y[train_ids, true]
          test_x = x[test_ids, feature_ids]
          test_y = y.shape[1].nil? ? y[test_ids] : y[test_ids, true]
          # Fit the estimator.
          start_time = Time.now.to_i
          @estimator.fit(train_x, train_y)
          # Calculate scores and prepare the report.
          report[:fit_time].push(Time.now.to_i - start_time)
          if @evaluator.nil?
            report[:test_score].push(@estimator.score(test_x, test_y))
            report[:train_score].push(@estimator.score(train_x, train_y)) if @return_train_score
          elsif log_loss?
            report[:test_score].push(@evaluator.score(test_y, @estimator.predict_proba(test_x)))
            if @return_train_score
              report[:train_score].push(@evaluator.score(train_y,
                                                         @estimator.predict_proba(train_x)))
            end
          else
            report[:test_score].push(@evaluator.score(test_y, @estimator.predict(test_x)))
            report[:train_score].push(@evaluator.score(train_y, @estimator.predict(train_x))) if @return_train_score
          end
        end
        report
      end

      private

      def kernel_machine?
        @estimator.class.name.include?('Rumale::KernelMachine')
      end

      def log_loss?
        @evaluator.is_a?(::Rumale::EvaluationMeasure::LogLoss)
      end
    end
  end
end
