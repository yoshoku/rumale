# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'
require 'svmkit/probabilistic_output'

module SVMKit
  # This module consists of the classes that implement generalized linear models.
  module LinearModel
    # SVC is a class that implements Support Vector Classifier
    # with stochastic gradient descent (SGD) optimization.
    # For multiclass classification problem, it uses one-vs-the-rest strategy.
    #
    # @example
    #   estimator =
    #     SVMKit::LinearModel::SVC.new(reg_param: 1.0, max_iter: 100, batch_size: 20, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # 1. S. Shalev-Shwartz and Y. Singer, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Proc. ICML'07, pp. 807--814, 2007.
    class SVC
      include Base::BaseEstimator
      include Base::Classifier

      # Return the weight vector for SVC.
      # @return [Numo::DFloat] (shape: [n_classes, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept) for SVC.
      # @return [Numo::DFloat] (shape: [n_classes])
      attr_reader :bias_term

      # Return the class labels.
      # @return [Numo::Int32] (shape: [n_classes])
      attr_reader :classes

      # Return the random generator for performing random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with Support Vector Machine by the SGD optimization.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param probability [Boolean] The flag indicating whether to perform probability estimation.
      # @param normalize [Boolean] The flag indicating whether to normalize the weight vector.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, fit_bias: false, bias_scale: 1.0,
                     max_iter: 100, batch_size: 50, probability: false, normalize: true, random_seed: nil)
        SVMKit::Validation.check_params_float(reg_param: reg_param, bias_scale: bias_scale)
        SVMKit::Validation.check_params_integer(max_iter: max_iter, batch_size: batch_size)
        SVMKit::Validation.check_params_boolean(fit_bias: fit_bias, probability: probability, normalize: normalize)
        SVMKit::Validation.check_params_type_or_nil(Integer, random_seed: random_seed)
        SVMKit::Validation.check_params_positive(reg_param: reg_param, bias_scale: bias_scale, max_iter: max_iter,
                                                 batch_size: batch_size)
        @params = {}
        @params[:reg_param] = reg_param
        @params[:fit_bias] = fit_bias
        @params[:bias_scale] = bias_scale
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:probability] = probability
        @params[:normalize] = normalize
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @weight_vec = nil
        @bias_term = nil
        @prob_param = nil
        @classes = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [SVC] The learned classifier itself.
      def fit(x, y)
        SVMKit::Validation.check_sample_array(x)
        SVMKit::Validation.check_label_array(y)
        SVMKit::Validation.check_sample_label_size(x, y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]
        n_classes = @classes.size
        _n_samples, n_features = x.shape

        if n_classes > 2
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @bias_term = Numo::DFloat.zeros(n_classes)
          @prob_param = Numo::DFloat.zeros(n_classes, 2)
          n_classes.times do |n|
            bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
            weight, bias = binary_fit(x, bin_y)
            @weight_vec[n, true] = weight
            @bias_term[n] = bias
            @prob_param[n, true] = if @params[:probability]
                                     SVMKit::ProbabilisticOutput.fit_sigmoid(x.dot(weight.transpose) + bias, bin_y)
                                   else
                                     Numo::DFloat[1, 0]
                                   end
          end
        else
          negative_label = y.to_a.uniq.sort.first
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec, @bias_term = binary_fit(x, bin_y)
          @prob_param = if @params[:probability]
                          SVMKit::ProbabilisticOutput.fit_sigmoid(x.dot(@weight_vec.transpose) + @bias_term, bin_y)
                        else
                          Numo::DFloat[1, 0]
                        end
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

        return Numo::Int32.cast(decision_function(x).ge(0.0)) * 2 - 1 if @classes.size <= 2

        n_samples, = x.shape
        decision_values = decision_function(x)
        Numo::Int32.asarray(Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] })
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        SVMKit::Validation.check_sample_array(x)

        if @classes.size > 2
          probs = 1.0 / (Numo::NMath.exp(@prob_param[true, 0] * decision_function(x) + @prob_param[true, 1]) + 1.0)
          return (probs.transpose / probs.sum(axis: 1)).transpose
        end

        n_samples, = x.shape
        probs = Numo::DFloat.zeros(n_samples, 2)
        probs[true, 1] = 1.0 / (Numo::NMath.exp(@prob_param[0] * decision_function(x) + @prob_param[1]) + 1.0)
        probs[true, 0] = 1.0 - probs[true, 1]
        probs
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about SVC.
      def marshal_dump
        { params: @params,
          weight_vec: @weight_vec,
          bias_term: @bias_term,
          prob_param: @prob_param,
          classes: @classes,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @weight_vec = obj[:weight_vec]
        @bias_term = obj[:bias_term]
        @prob_param = obj[:prob_param]
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
          sub_samples = samples[subset_ids, true]
          sub_bin_y = bin_y[subset_ids]
          target_ids = (sub_samples.dot(weight_vec.transpose) * sub_bin_y).lt(1.0).where
          n_targets = target_ids.size
          next if n_targets.zero?
          # update the weight vector.
          mean_vec = sub_samples[target_ids, true].transpose.dot(sub_bin_y[target_ids]) / n_targets
          weight_vec -= learning_rate(t) * (@params[:reg_param] * weight_vec - mean_vec)
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
