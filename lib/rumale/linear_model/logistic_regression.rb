# frozen_string_literal: true

require 'rumale/linear_model/base_linear_model'
require 'rumale/base/classifier'

module Rumale
  module LinearModel
    # LogisticRegression is a class that implements Logistic Regression
    # with mini-batch stochastic gradient descent optimization.
    # For multiclass classification problem, it uses one-vs-the-rest strategy.
    #
    # Rumale::SVM provides Logistic Regression based on LIBLINEAR.
    # If you prefer execution speed, you should use Rumale::SVM::LogisticRegression.
    # https://github.com/yoshoku/rumale-svm
    #
    # @example
    #   estimator =
    #     Rumale::LinearModel::LogisticRegression.new(reg_param: 1.0, max_iter: 1000, batch_size: 20, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Mathematical Programming, vol. 127 (1), pp. 3--30, 2011.
    class LogisticRegression < BaseLinearModel
      include Base::Classifier

      # Return the weight vector for Logistic Regression.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept) for Logistic Regression.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :bias_term

      # Return the class labels.
      # @return [Numo::Int32] (shape: [n_classes])
      attr_reader :classes

      # Return the random generator for performing random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with Logisitc Regression by the SGD optimization.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      #   If fit_bias is true, the feature vector v becoms [v; bias_scale].
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param optimizer [Optimizer] The optimizer to calculate adaptive learning rate.
      #   If nil is given, Nadam is used.
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, fit_bias: false, bias_scale: 1.0,
                     max_iter: 1000, batch_size: 20, optimizer: nil, n_jobs: nil, random_seed: nil)
        check_params_float(reg_param: reg_param, bias_scale: bias_scale)
        check_params_integer(max_iter: max_iter, batch_size: batch_size)
        check_params_boolean(fit_bias: fit_bias)
        check_params_type_or_nil(Integer, n_jobs: n_jobs, random_seed: random_seed)
        check_params_positive(reg_param: reg_param, bias_scale: bias_scale, max_iter: max_iter, batch_size: batch_size)
        super
        @classes = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [LogisticRegression] The learned classifier itself.
      def fit(x, y)
        check_sample_array(x)
        check_label_array(y)
        check_sample_label_size(x, y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]

        if multiclass_problem?
          n_classes = @classes.size
          n_features = x.shape[1]
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @bias_term = Numo::DFloat.zeros(n_classes)
          if enable_parallel?
            # :nocov:
            models = parallel_map(n_classes) do |n|
              bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
              partial_fit(x, bin_y)
            end
            # :nocov:
            n_classes.times { |n| @weight_vec[n, true], @bias_term[n] = models[n] }
          else
            n_classes.times do |n|
              bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
              @weight_vec[n, true], @bias_term[n] = partial_fit(x, bin_y)
            end
          end
        else
          negative_label = @classes[0]
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec, @bias_term = partial_fit(x, bin_y)
        end

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        check_sample_array(x)
        x.dot(@weight_vec.transpose) + @bias_term
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        check_sample_array(x)

        n_samples, = x.shape
        decision_values = predict_proba(x)
        predicted = if enable_parallel?
                      parallel_map(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                    else
                      Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                    end
        Numo::Int32.asarray(predicted)
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        check_sample_array(x)

        proba = 1.0 / (Numo::NMath.exp(-decision_function(x)) + 1.0)
        return (proba.transpose / proba.sum(axis: 1)).transpose.dup if multiclass_problem?

        n_samples, = x.shape
        probs = Numo::DFloat.zeros(n_samples, 2)
        probs[true, 1] = proba
        probs[true, 0] = 1.0 - proba
        probs
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about LogisticRegression.
      def marshal_dump
        { params: @params,
          weight_vec: @weight_vec,
          bias_term: @bias_term,
          classes: @classes,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @weight_vec = obj[:weight_vec]
        @bias_term = obj[:bias_term]
        @classes = obj[:classes]
        @rng = obj[:rng]
        nil
      end

      private

      def calc_loss_gradient(x, y, weight)
        y / (Numo::NMath.exp(-y * x.dot(weight)) + 1.0) - y
      end

      def multiclass_problem?
        @classes.size > 2
      end
    end
  end
end
