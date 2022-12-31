# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/estimator'
require 'rumale/base/regressor'

module Rumale
  module Ensemble
    # VotingRegressor is a class that implements regressor with voting ensemble method.
    #
    # @example
    #   require 'rumale/ensemble/voting_regressor'
    #
    #   estimators = {
    #     rdg: Rumale::LinearModel::Ridge.new(reg_param: 1e-2, random_seed: 1),
    #     mlp: Rumale::NeuralNetwork::MLPRegressor.new(hidden_units: [256], random_seed: 1),
    #     rnd: Rumale::Ensemble::RandomForestRegressor.new(random_seed: 1)
    #   }
    #   weights = { rdg: 0.2, mlp: 0.3, rnd: 0.5 }
    #
    #   regressor = Rumale::Ensemble::VotingRegressor.new(estimators: estimators, weights: weights, voting: 'soft')
    #   regressor.fit(x_train, y_train)
    #   results = regressor.predict(x_test)
    #
    # *Reference*
    # - Zhou, Z-H., "Ensemble Methods - Foundations and Algorithms," CRC Press Taylor and Francis Group, Chapman and Hall/CRC, 2012.
    class VotingRegressor < ::Rumale::Base::Estimator
      include ::Rumale::Base::Regressor

      # Return the sub-regressors that voted.
      # @return [Hash<Symbol,Regressor>]
      attr_reader :estimators

      # Create a new ensembled regressor with voting rule.
      #
      # @param estimators [Hash<Symbol,Regressor>] The sub-regressors to vote.
      # @param weights [Hash<Symbol,Float>] The weight value for each regressor.
      def initialize(estimators:, weights: nil)
        super()
        @estimators = estimators
        @params = {
          weights: weights || estimators.each_key.with_object({}) { |name, w| w[name] = 1.0 }
        }
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [VotingRegressor] The learned regressor itself.
      def fit(x, y)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        y = ::Rumale::Validation.check_convert_target_value_array(y)
        ::Rumale::Validation.check_sample_size(x, y)

        @n_outputs = y.ndim > 1 ? y.shape[1] : 1
        @estimators.each_key { |name| @estimators[name].fit(x, y) }

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted value per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        z = single_target? ? Numo::DFloat.zeros(x.shape[0]) : Numo::DFloat.zeros(x.shape[0], @n_outputs)
        sum_weight = @params[:weights].each_value.sum
        @estimators.each do |name, estimator|
          z += @params[:weights][name] * estimator.predict(x)
        end
        z / sum_weight
      end

      private

      def single_target?
        @n_outputs == 1
      end
    end
  end
end
