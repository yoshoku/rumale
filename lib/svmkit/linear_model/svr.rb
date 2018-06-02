# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/regressor'
require 'svmkit/optimizer/nadam'

module SVMKit
  module LinearModel
    # SVR is a class that implements Support Vector Regressor
    # with mini-batch stochastic gradient descent optimization.
    #
    # @example
    #   estimator =
    #     SVMKit::LinearModel::SVR.new(reg_param: 1.0, epsilon: 0.1, max_iter: 1000, batch_size: 20, random_seed: 1)
    #   estimator.fit(training_samples, traininig_target_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # 1. S. Shalev-Shwartz and Y. Singer, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Proc. ICML'07, pp. 807--814, 2007.
    class SVR
      include Base::BaseEstimator
      include Base::Regressor
      include Validation

      # Return the weight vector for SVR.
      # @return [Numo::DFloat] (shape: [n_outputs, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept) for SVR.
      # @return [Numo::DFloat] (shape: [n_outputs])
      attr_reader :bias_term

      # Return the random generator for performing random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new regressor with Support Vector Machine by the SGD optimization.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param epsilon [Float] The margin of tolerance.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param optimizer [Optimizer] The optimizer to calculate adaptive learning rate.
      #   Nadam is selected automatically on current version.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, fit_bias: false, bias_scale: 1.0, epsilon: 0.1,
                     max_iter: 1000, batch_size: 20, optimizer: nil, random_seed: nil)
        check_params_float(reg_param: reg_param, bias_scale: bias_scale, epsilon: epsilon)
        check_params_integer(max_iter: max_iter, batch_size: batch_size)
        check_params_boolean(fit_bias: fit_bias)
        check_params_type_or_nil(Integer, random_seed: random_seed)
        check_params_positive(reg_param: reg_param, bias_scale: bias_scale, epsilon: epsilon,
                              max_iter: max_iter, batch_size: batch_size)
        @params = {}
        @params[:reg_param] = reg_param
        @params[:fit_bias] = fit_bias
        @params[:bias_scale] = bias_scale
        @params[:epsilon] = epsilon
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:optimizer] = optimizer
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @weight_vec = nil
        @bias_term = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [SVR] The learned regressor itself.
      def fit(x, y)
        check_sample_array(x)
        check_tvalue_array(y)
        check_sample_tvalue_size(x, y)

        n_outputs = y.shape[1].nil? ? 1 : y.shape[1]
        _n_samples, n_features = x.shape

        if n_outputs > 1
          @weight_vec = Numo::DFloat.zeros(n_outputs, n_features)
          @bias_term = Numo::DFloat.zeros(n_outputs)
          n_outputs.times do |n|
            weight, bias = single_fit(x, y[true, n])
            @weight_vec[n, true] = weight
            @bias_term[n] = bias
          end
        else
          @weight_vec, @bias_term = single_fit(x, y)
        end

        self
      end

      # Predict values for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the values.
      # @return [Numo::DFloat] (shape: [n_samples, n_outputs]) Predicted values per sample.
      def predict(x)
        check_sample_array(x)
        x.dot(@weight_vec.transpose) + @bias_term
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about SVR.
      def marshal_dump
        { params: @params,
          weight_vec: @weight_vec,
          bias_term: @bias_term,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @weight_vec = obj[:weight_vec]
        @bias_term = obj[:bias_term]
        @rng = obj[:rng]
        nil
      end

      private

      def single_fit(x, y)
        # Expand feature vectors for bias term.
        samples = @params[:fit_bias] ? expand_feature(x) : x
        # Initialize some variables.
        n_samples, n_features = samples.shape
        rand_ids = [*0...n_samples].shuffle(random: @rng)
        weight_vec = Numo::DFloat.zeros(n_features)
        optimizer = Optimizer::Nadam.new
        # Start optimization.
        @params[:max_iter].times do |_t|
          # random sampling
          subset_ids = rand_ids.shift(@params[:batch_size])
          rand_ids.concat(subset_ids)
          data = samples[subset_ids, true]
          values = y[subset_ids]
          # update the weight vector.
          loss_grad = loss_gradient(data, values, weight_vec)
          weight_vec = optimizer.call(weight_vec, weight_gradient(loss_grad, data, weight_vec))
        end
        split_weight_vec_bias(weight_vec)
      end

      def loss_gradient(x, y, weight)
        z = x.dot(weight)
        grad = Numo::DFloat.zeros(@params[:batch_size])
        grad[(z - y).gt(@params[:epsilon]).where] = 1
        grad[(y - z).gt(@params[:epsilon]).where] = -1
        grad
      end

      def weight_gradient(loss_grad, x, weight)
        x.transpose.dot(loss_grad) / @params[:batch_size] + @params[:reg_param] * weight
      end

      def expand_feature(x)
        Numo::NArray.hstack([x, Numo::DFloat.ones([x.shape[0], 1]) * @params[:bias_scale]])
      end

      def split_weight_vec_bias(weight_vec)
        weights = @params[:fit_bias] ? weight_vec[0...-1] : weight_vec
        bias = @params[:fit_bias] ? weight_vec[-1] : 0.0
        [weights, bias]
      end
    end
  end
end
