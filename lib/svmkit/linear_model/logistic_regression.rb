# frozen_string_literal: true

require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

module SVMKit
  # This module consists of the classes that implement generalized linear models.
  module LinearModel
    # LogisticRegression is a class that implements Logistic Regression
    # with stochastic gradient descent (SGD) optimization.
    # For multiclass classification problem, it uses one-vs-the-rest strategy.
    #
    # @example
    #   estimator =
    #     SVMKit::LinearModel::LogisticRegression.new(reg_param: 1.0, max_iter: 100, batch_size: 20, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # 1. S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Mathematical Programming, vol. 127 (1), pp. 3--30, 2011.
    class LogisticRegression
      include Base::BaseEstimator
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
      # @param normalize [Boolean] The flag indicating whether to normalize the weight vector.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, fit_bias: false, bias_scale: 1.0,
                     max_iter: 100, batch_size: 50, normalize: true, random_seed: nil)
        SVMKit::Validation.check_params_float(reg_param: reg_param, bias_scale: bias_scale)
        SVMKit::Validation.check_params_integer(max_iter: max_iter, batch_size: batch_size)
        SVMKit::Validation.check_params_boolean(fit_bias: fit_bias, normalize: normalize)
        SVMKit::Validation.check_params_type_or_nil(Integer, random_seed: random_seed)
        SVMKit::Validation.check_params_positive(reg_param: reg_param, bias_scale: bias_scale, max_iter: max_iter,
                                                 batch_size: batch_size)
        @params = {}
        @params[:reg_param] = reg_param
        @params[:fit_bias] = fit_bias
        @params[:bias_scale] = bias_scale
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:normalize] = normalize
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @weight_vec = nil
        @bias_term = nil
        @classes = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [LogisticRegression] The learned classifier itself.
      def fit(x, y)
        SVMKit::Validation.check_sample_array(x)
        SVMKit::Validation.check_label_array(y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]
        n_classes = @classes.size
        _n_samples, n_features = x.shape

        if n_classes > 2
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @bias_term = Numo::DFloat.zeros(n_classes)
          n_classes.times do |n|
            bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
            weight, bias = binary_fit(x, bin_y)
            @weight_vec[n, true] = weight
            @bias_term[n] = bias
          end
        else
          negative_label = y.to_a.uniq.sort.first
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec, @bias_term = binary_fit(x, bin_y)
        end

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        SVMKit::Validation.check_sample_array(x)

        x.dot(@weight_vec.transpose) + @bias_term
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        SVMKit::Validation.check_sample_array(x)

        return Numo::Int32.cast(predict_proba(x)[true, 1].ge(0.5)) * 2 - 1 if @classes.size <= 2

        n_samples, = x.shape
        decision_values = predict_proba(x)
        Numo::Int32.asarray(Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] })
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        SVMKit::Validation.check_sample_array(x)

        proba = 1.0 / (Numo::NMath.exp(-decision_function(x)) + 1.0)
        return (proba.transpose / proba.sum(axis: 1)).transpose if @classes.size > 2

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

      def binary_fit(x, bin_y)
        # Expand feature vectors for bias term.
        samples = @params[:fit_bias] ? expand_feature(x) : x
        # Initialize some variables.
        n_samples, n_features = samples.shape
        rand_ids = [*0...n_samples].shuffle(random: @rng)
        weight_vec = Numo::DFloat.zeros(n_features)
        # Start optimization.
        @params[:max_iter].times do |t|
          # random sampling
          subset_ids = rand_ids.shift(@params[:batch_size])
          rand_ids.concat(subset_ids)
          # update the weight vector.
          df = samples[subset_ids, true].dot(weight_vec.transpose)
          coef = bin_y[subset_ids] / (Numo::NMath.exp(-bin_y[subset_ids] * df) + 1.0) - bin_y[subset_ids]
          mean_vec = samples[subset_ids, true].transpose.dot(coef) / @params[:batch_size]
          weight_vec -= learning_rate(t) * (@params[:reg_param] * weight_vec + mean_vec)
          # scale the weight vector.
          normalize_weight_vec(weight_vec) if @params[:normalize]
        end
        split_weight_vec_bias(weight_vec)
      end

      def expand_feature(x)
        Numo::NArray.hstack([x, Numo::DFloat.ones([x.shape[0], 1]) * @params[:bias_scale]])
      end

      def learning_rate(iter)
        1.0 / (@params[:reg_param] * (iter + 1))
      end

      def normalize_weight_vec(weight_vec)
        norm = Math.sqrt(weight_vec.dot(weight_vec))
        weight_vec * [1.0, (1.0 / @params[:reg_param]**0.5) / (norm + 1.0e-12)].min
      end

      def split_weight_vec_bias(weight_vec)
        weights = @params[:fit_bias] ? weight_vec[0...-1] : weight_vec
        bias = @params[:fit_bias] ? weight_vec[-1] : 0.0
        [weights, bias]
      end
    end
  end
end
