# frozen_string_literal: true

require 'rumale/validation'
require 'rumale/base/base_estimator'

module Rumale
  # Module implements utilities of pipeline that cosists of a chain of transfomers and estimators.
  module Pipeline
    # Pipeline is a class that implements the function to perform the transformers and estimators sequencially.
    #
    # @example
    #   rbf = Rumale::KernelApproximation::RBF.new(gamma: 1.0, n_coponents: 128, random_seed: 1)
    #   svc = Rumale::LinearModel::SVC.new(reg_param: 1.0, fit_bias: true, max_iter: 5000, random_seed: 1)
    #   pipeline = Rumale::Pipeline::Pipeline.new(steps: { trs: rbf, est: svc })
    #   pipeline.fit(training_samples, traininig_labels)
    #   results = pipeline.predict(testing_samples)
    #
    class Pipeline
      include Base::BaseEstimator
      include Validation

      # Return the steps.
      # @return [Hash]
      attr_reader :steps

      # Create a new pipeline.
      #
      # @param steps [Hash] List of transformers and estimators. The order of transforms follows the insertion order of hash keys.
      #   The last entry is considered an estimator.
      def initialize(steps:)
        check_params_type(Hash, steps: steps)
        validate_steps(steps)
        @params = {}
        @steps = steps
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be transformed and used for fitting the model.
      # @param y [Numo::NArray] (shape: [n_samples, n_outputs]) The target values or labels to be used for fitting the model.
      # @return [Pipeline] The learned pipeline itself.
      def fit(x, y)
        check_sample_array(x)
        trans_x = apply_transforms(x, y, fit: true)
        last_estimator&.fit(trans_x, y)
        self
      end

      # Call the fit_predict method of last estimator after applying all transforms.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be transformed and used for fitting the model.
      # @param y [Numo::NArray] (shape: [n_samples, n_outputs], default: nil) The target values or labels to be used for fitting the model.
      # @return [Numo::NArray] The predicted results by last estimator.
      def fit_predict(x, y = nil)
        check_sample_array(x)
        trans_x = apply_transforms(x, y, fit: true)
        last_estimator.fit_predict(trans_x)
      end

      # Call the fit_transform method of last estimator after applying all transforms.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be transformed and used for fitting the model.
      # @param y [Numo::NArray] (shape: [n_samples, n_outputs], default: nil) The target values or labels to be used for fitting the model.
      # @return [Numo::NArray] The predicted results by last estimator.
      def fit_transform(x, y = nil)
        check_sample_array(x)
        trans_x = apply_transforms(x, y, fit: true)
        last_estimator.fit_transform(trans_x, y)
      end

      # Call the decision_function method of last estimator after applying all transforms.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples]) Confidence score per sample.
      def decision_function(x)
        check_sample_array(x)
        trans_x = apply_transforms(x)
        last_estimator.decision_function(trans_x)
      end

      # Call the predict method of last estimator after applying all transforms.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to obtain prediction result.
      # @return [Numo::NArray] The predicted results by last estimator.
      def predict(x)
        check_sample_array(x)
        trans_x = apply_transforms(x)
        last_estimator.predict(trans_x)
      end

      # Call the predict_log_proba method of last estimator after applying all transforms.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the log-probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted log-probability of each class per sample.
      def predict_log_proba(x)
        check_sample_array(x)
        trans_x = apply_transforms(x)
        last_estimator.predict_log_proba(trans_x)
      end

      # Call the predict_proba method of last estimator after applying all transforms.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        check_sample_array(x)
        trans_x = apply_transforms(x)
        last_estimator.predict_proba(trans_x)
      end

      # Call the transform method of last estimator after applying all transforms.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to be transformed.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed samples.
      def transform(x)
        check_sample_array(x)
        trans_x = apply_transforms(x)
        last_estimator.nil? ? trans_x : last_estimator.transform(trans_x)
      end

      # Call the inverse_transform method in reverse order.
      #
      # @param z [Numo::DFloat] (shape: [n_samples, n_components]) The transformed samples to be restored into original space.
      # @return [Numo::DFloat] (shape: [n_samples, n_featuress]) The restored samples.
      def inverse_transform(z)
        check_sample_array(z)
        itrans_z = z
        @steps.keys.reverse_each do |name|
          transformer = @steps[name]
          next if transformer.nil?
          itrans_z = transformer.inverse_transform(itrans_z)
        end
        itrans_z
      end

      # Call the score method of last estimator after applying all transforms.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) Testing data.
      # @param y [Numo::NArray] (shape: [n_samples, n_outputs]) True target values or labels for testing data.
      # @return [Float] The score of last estimator
      def score(x, y)
        check_sample_array(x)
        trans_x = apply_transforms(x)
        last_estimator.score(trans_x, y)
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about Pipeline.
      def marshal_dump
        { params: @params,
          steps: @steps }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @steps = obj[:steps]
        nil
      end

      private

      def validate_steps(steps)
        steps.keys[0...-1].each do |name|
          transformer = steps[name]
          next if transformer.nil? || %i[fit transform].all? { |m| transformer.class.method_defined?(m) }
          raise TypeError,
                'Class of intermediate step in pipeline should be implemented fit and transform methods: ' \
                "#{name} => #{transformer.class}"
        end

        estimator = steps[steps.keys.last]
        unless estimator.nil? || estimator.class.method_defined?(:fit) # rubocop:disable Style/GuardClause
          raise TypeError,
                'Class of last step in pipeline should be implemented fit method: ' \
                "#{steps.keys.last} => #{estimator.class}"
        end
      end

      def apply_transforms(x, y = nil, fit: false)
        trans_x = x
        @steps.keys[0...-1].each do |name|
          transformer = @steps[name]
          next if transformer.nil?
          transformer.fit(trans_x, y) if fit
          trans_x = transformer.transform(trans_x)
        end
        trans_x
      end

      def last_estimator
        @steps[@steps.keys.last]
      end
    end
  end
end
