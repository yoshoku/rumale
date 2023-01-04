# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/model_selection/cross_validation'

module Rumale
  module ModelSelection
    # GridSearchCV is a class that performs hyperparameter optimization with grid search method.
    #
    # @example
    #   require 'rumale/ensemble'
    #   require 'rumale/model_selection/stratified_k_fold'
    #   require 'rumale/model_selection/grid_search_cv'
    #
    #   rfc = Rumale::Ensemble::RandomForestClassifier.new(random_seed: 1)
    #   pg = { n_estimators: [5, 10], max_depth: [3, 5], max_leaf_nodes: [15, 31] }
    #   kf = Rumale::ModelSelection::StratifiedKFold.new(n_splits: 5)
    #   gs = Rumale::ModelSelection::GridSearchCV.new(estimator: rfc, param_grid: pg, splitter: kf)
    #   gs.fit(samples, labels)
    #   p gs.cv_results
    #   p gs.best_params
    #
    # @example
    #   rbf = Rumale::KernelApproximation::RBF.new(random_seed: 1)
    #   svc = Rumale::LinearModel::SVC.new
    #   pipe = Rumale::Pipeline::Pipeline.new(steps: { rbf: rbf, svc: svc })
    #   pg = { rbf__gamma: [32.0, 1.0], rbf__n_components: [4, 128], svc__reg_param: [16.0, 0.1] }
    #   kf = Rumale::ModelSelection::StratifiedKFold.new(n_splits: 5)
    #   gs = Rumale::ModelSelection::GridSearchCV.new(estimator: pipe, param_grid: pg, splitter: kf)
    #   gs.fit(samples, labels)
    #   p gs.cv_results
    #   p gs.best_params
    #
    class GridSearchCV < ::Rumale::Base::Estimator
      # Return the result of cross validation for each parameter.
      # @return [Hash]
      attr_reader :cv_results

      # Return the score of the estimator learned with the best parameter.
      # @return [Float]
      attr_reader :best_score

      # Return the best parameter set.
      # @return [Hash]
      attr_reader :best_params

      # Return the index of the best parameter.
      # @return [Integer]
      attr_reader :best_index

      # Return the estimator learned with the best parameter.
      # @return [Estimator]
      attr_reader :best_estimator

      # Create a new grid search method.
      #
      # @param estimator [Classifier/Regresor] The estimator to be searched for optimal parameters with grid search method.
      # @param param_grid [Array<Hash>] The parameter sets is represented with array of hash that
      #   consists of parameter names as keys and array of parameter values as values.
      # @param splitter [Splitter] The splitter that divides dataset to training and testing dataset on cross validation.
      # @param evaluator [Evaluator] The evaluator that calculates score of estimator results on cross validation.
      #   If nil is given, the score method of estimator is used to evaluation.
      # @param greater_is_better [Boolean] The flag that indicates whether the estimator is better as
      #   evaluation score is larger.
      def initialize(estimator: nil, param_grid: nil, splitter: nil, evaluator: nil, greater_is_better: true)
        super()
        @params = {
          param_grid: valid_param_grid(param_grid),
          estimator: Marshal.load(Marshal.dump(estimator)),
          splitter: Marshal.load(Marshal.dump(splitter)),
          evaluator: Marshal.load(Marshal.dump(evaluator)),
          greater_is_better: greater_is_better
        }
      end

      # Fit the model with given training data and all sets of parameters.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::NArray] (shape: [n_samples, n_outputs]) The target values or labels to be used for fitting the model.
      # @return [GridSearchCV] The learned estimator with grid search.
      def fit(x, y)
        init_attrs

        param_combinations.each do |prm_set|
          prm_set.each do |prms|
            report = perform_cross_validation(x, y, prms)
            store_cv_result(prms, report)
          end
        end

        find_best_params

        @best_estimator = configurated_estimator(@best_params)
        @best_estimator.fit(x, y)
        self
      end

      # Call the decision_function method of learned estimator with the best parameter.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples]) Confidence score per sample.
      def decision_function(x)
        @best_estimator.decision_function(x)
      end

      # Call the predict method of learned estimator with the best parameter.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to obtain prediction result.
      # @return [Numo::NArray] Predicted results.
      def predict(x)
        @best_estimator.predict(x)
      end

      # Call the predict_log_proba method of learned estimator with the best parameter.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the log-probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted log-probability of each class per sample.
      def predict_log_proba(x)
        @best_estimator.predict_log_proba(x)
      end

      # Call the predict_proba method of learned estimator with the best parameter.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        @best_estimator.predict_proba(x)
      end

      # Call the score method of learned estimator with the best parameter.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) Testing data.
      # @param y [Numo::NArray] (shape: [n_samples, n_outputs]) True target values or labels for testing data.
      # @return [Float] The score of estimator.
      def score(x, y)
        @best_estimator.score(x, y)
      end

      private

      def valid_param_grid(grid)
        raise TypeError, 'Expect class of param_grid to be Hash or Array' unless grid.is_a?(Hash) || grid.is_a?(Array)

        grid = [grid] if grid.is_a?(Hash)
        grid.each do |h|
          raise TypeError, 'Expect class of elements in param_grid to be Hash' unless h.is_a?(Hash)
          raise TypeError, 'Expect class of parameter values in param_grid to be Array' unless h.values.all?(Array)
        end
        grid
      end

      def param_combinations
        @param_combinations ||= @params[:param_grid].map do |prm|
          x = prm.sort.to_h.map { |k, v| [k].product(v) }
          x[0].product(*x[1...x.size]).map(&:to_h)
        end
      end

      def perform_cross_validation(x, y, prms)
        est = configurated_estimator(prms)
        cv = ::Rumale::ModelSelection::CrossValidation.new(estimator: est, splitter: @params[:splitter],
                                                           evaluator: @params[:evaluator], return_train_score: true)
        cv.perform(x, y)
      end

      def configurated_estimator(prms)
        estimator = Marshal.load(Marshal.dump(@params[:estimator]))
        if pipeline?
          prms.each do |k, v|
            est_name, prm_name = k.to_s.split('__')
            estimator.steps[est_name.to_sym].params[prm_name.to_sym] = v
          end
        else
          prms.each { |k, v| estimator.params[k] = v }
        end
        estimator
      end

      def init_attrs
        @cv_results = %i[mean_test_score std_test_score
                         mean_train_score std_train_score
                         mean_fit_time std_fit_time params].to_h { |v| [v, []] }
        @best_score = nil
        @best_params = nil
        @best_index = nil
        @best_estimator = nil
      end

      def store_cv_result(prms, report)
        test_scores = Numo::DFloat[*report[:test_score]]
        train_scores = Numo::DFloat[*report[:train_score]]
        fit_times = Numo::DFloat[*report[:fit_time]]
        @cv_results[:mean_test_score].push(test_scores.mean)
        @cv_results[:std_test_score].push(test_scores.stddev)
        @cv_results[:mean_train_score].push(train_scores.mean)
        @cv_results[:std_train_score].push(train_scores.stddev)
        @cv_results[:mean_fit_time].push(fit_times.mean)
        @cv_results[:std_fit_time].push(fit_times.stddev)
        @cv_results[:params].push(prms)
      end

      def find_best_params
        @best_score = @params[:greater_is_better] ? @cv_results[:mean_test_score].max : @cv_results[:mean_test_score].min
        @best_index = @cv_results[:mean_test_score].index(@best_score)
        @best_params = @cv_results[:params][@best_index]
      end

      def pipeline?
        @params[:estimator].class.name.include?('Rumale::Pipeline')
      end
    end
  end
end
