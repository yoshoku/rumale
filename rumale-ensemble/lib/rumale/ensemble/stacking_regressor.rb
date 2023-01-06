# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/estimator'
require 'rumale/base/regressor'
require 'rumale/linear_model/ridge'
require 'rumale/model_selection/k_fold'

module Rumale
  module Ensemble
    # StackingRegressor is a class that implements regressor with stacking method.
    #
    # @example
    #   require 'rumale/ensemble/stacking_regressor'
    #
    #   estimators = {
    #     las: Rumale::LinearModel::Lasso.new(reg_param: 1e-2, random_seed: 1),
    #     mlp: Rumale::NeuralNetwork::MLPRegressor.new(hidden_units: [256], random_seed: 1),
    #     rnd: Rumale::Ensemble::RandomForestRegressor.new(random_seed: 1)
    #   }
    #   meta_estimator = Rumale::LinearModel::Ridge.new
    #   regressor = Rumale::Ensemble::StackedRegressor.new(
    #     estimators: estimators, meta_estimator: meta_estimator, random_seed: 1
    #   )
    #   regressor.fit(training_samples, training_values)
    #   results = regressor.predict(testing_samples)
    #
    # *Reference*
    # - Zhou, Z-H., "Ensemble Methods - Foundations and Algorithms," CRC Press Taylor and Francis Group, Chapman and Hall/CRC, 2012.
    class StackingRegressor < ::Rumale::Base::Estimator
      include ::Rumale::Base::Regressor

      # Return the base regressors.
      # @return [Hash<Symbol,Regressor>]
      attr_reader :estimators

      # Return the meta regressor.
      # @return [Regressor]
      attr_reader :meta_estimator

      # Create a new regressor with stacking method.
      #
      # @param estimators [Hash<Symbol,Regressor>] The base regressors for extracting meta features.
      # @param meta_estimator [Regressor/Nil] The meta regressor that predicts values.
      #   If nil is given, Ridge is used.
      # @param n_splits [Integer] The number of folds for cross validation with k-fold on meta feature extraction in training phase.
      # @param shuffle [Boolean] The flag indicating whether to shuffle the dataset on cross validation.
      # @param passthrough [Boolean] The flag indicating whether to concatenate the original features and meta features when training the meta regressor.
      # @param random_seed [Integer/Nil] The seed value using to initialize the random generator on cross validation.
      def initialize(estimators:, meta_estimator: nil, n_splits: 5, shuffle: true, passthrough: false, random_seed: nil)
        super()
        @estimators = estimators
        @meta_estimator = meta_estimator || ::Rumale::LinearModel::Ridge.new
        @params = {
          n_splits: n_splits,
          shuffle: shuffle,
          passthrough: passthrough,
          random_seed: random_seed || srand
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target variables to be used for fitting the model.
      # @return [StackedRegressor] The learned regressor itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        n_samples, n_features = x.shape
        n_outputs = y.ndim == 1 ? 1 : y.shape[1]

        # training base regressors with all training data.
        @estimators.each_key { |name| @estimators[name].fit(x, y) }

        # detecting size of output for each base regressor.
        @output_size = detect_output_size(n_features)

        # extracting meta features with base regressors.
        n_components = @output_size.values.sum
        z = Numo::DFloat.zeros(n_samples, n_components)

        kf = ::Rumale::ModelSelection::KFold.new(
          n_splits: @params[:n_splits], shuffle: @params[:shuffle], random_seed: @params[:random_seed]
        )

        kf.split(x, y).each do |train_ids, valid_ids|
          x_train = x[train_ids, true]
          y_train = n_outputs == 1 ? y[train_ids] : y[train_ids, true]
          x_valid = x[valid_ids, true]
          f_start = 0
          @estimators.each_key do |name|
            est_fold = Marshal.load(Marshal.dump(@estimators[name]))
            f_last = f_start + @output_size[name]
            f_position = @output_size[name] == 1 ? f_start : f_start...f_last
            z[valid_ids, f_position] = est_fold.fit(x_train, y_train).predict(x_valid)
            f_start = f_last
          end
        end

        # concatenating original features.
        z = Numo::NArray.hstack([z, x]) if @params[:passthrough]

        # training meta regressor.
        @meta_estimator.fit(z, y)

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) The predicted values per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        z = transform(x)
        @meta_estimator.predict(z)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The meta features for samples.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        n_samples = x.shape[0]
        n_components = @output_size.values.sum
        z = Numo::DFloat.zeros(n_samples, n_components)
        f_start = 0
        @estimators.each_key do |name|
          f_last = f_start + @output_size[name]
          f_position = @output_size[name] == 1 ? f_start : f_start...f_last
          z[true, f_position] = @estimators[name].predict(x)
          f_start = f_last
        end
        z = Numo::NArray.hstack([z, x]) if @params[:passthrough]
        z
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target variables to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The meta features for training data.
      def fit_transform(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        fit(x, y).transform(x)
      end

      private

      def detect_output_size(n_features)
        x_dummy = Numo::DFloat.new(2, n_features).rand
        @estimators.each_key.with_object({}) do |name, obj|
          output_dummy = @estimators[name].predict(x_dummy)
          obj[name] = output_dummy.ndim == 1 ? 1 : output_dummy.shape[1]
        end
      end
    end
  end
end
