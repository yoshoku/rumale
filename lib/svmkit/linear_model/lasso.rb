# frozen_string_literal: true

require 'svmkit/validation'
require 'svmkit/base/base_estimator'
require 'svmkit/base/regressor'

module SVMKit
  module LinearModel
    # Lasso is a class that implements Lasso Regression
    # with stochastic gradient descent (SGD) optimization.
    #
    # @example
    #   estimator =
    #     SVMKit::LinearModel::Lasso.new(reg_param: 0.1, max_iter: 5000, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_values)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Shalev-Shwartz and Y. Singer, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Proc. ICML'07, pp. 807--814, 2007.
    # - L. Bottou, "Large-Scale Machine Learning with Stochastic Gradient Descent," Proc. COMPSTAT'10, pp. 177--186, 2010.
    # - I. Sutskever, J. Martens, G. Dahl, and G. Hinton, "On the importance of initialization and momentum in deep learning," Proc. ICML'13, pp. 1139--1147, 2013.
    # - G. Hinton, N. Srivastava, and K. Swersky, "Lecture 6e rmsprop," Neural Networks for Machine Learning, 2012.
    class Lasso
      include Base::BaseEstimator
      include Base::Regressor
      include Validation

      # Return the weight vector.
      # @return [Numo::DFloat] (shape: [n_outputs, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept).
      # @return [Numo::DFloat] (shape: [n_outputs])
      attr_reader :bias_term

      # Return the random generator for random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new Lasso regressor.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param learning_rate [Float] The learning rate for optimization.
      # @param decay [Float] The discounting factor for RMS prop optimization.
      # @param momentum [Float] The momentum for optimization.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, fit_bias: false, learning_rate: 0.01, decay: 0.9, momentum: 0.9,
                     max_iter: 1000, batch_size: 10, random_seed: nil)
        check_params_float(reg_param: reg_param,
                           learning_rate: learning_rate, decay: decay, momentum: momentum)
        check_params_integer(max_iter: max_iter, batch_size: batch_size)
        check_params_boolean(fit_bias: fit_bias)
        check_params_type_or_nil(Integer, random_seed: random_seed)
        check_params_positive(reg_param: reg_param,
                              learning_rate: learning_rate, decay: decay, momentum: momentum,
                              max_iter: max_iter, batch_size: batch_size)
        @params = {}
        @params[:reg_param] = reg_param
        @params[:fit_bias] = fit_bias
        @params[:learning_rate] = learning_rate
        @params[:decay] = decay
        @params[:momentum] = momentum
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @weight_vec = nil
        @bias_term = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples, n_outputs]) The target values to be used for fitting the model.
      # @return [Lasso] The learned regressor itself.
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
      # @return [Hash] The marshal data about Lasso.
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
        left_weight_vec = Numo::DFloat.zeros(n_features)
        left_weight_sqrsum = Numo::DFloat.zeros(n_features)
        left_weight_update = Numo::DFloat.zeros(n_features)
        right_weight_vec = Numo::DFloat.zeros(n_features)
        right_weight_sqrsum = Numo::DFloat.zeros(n_features)
        right_weight_update = Numo::DFloat.zeros(n_features)
        # Start optimization.
        @params[:max_iter].times do |_t|
          # Random sampling.
          subset_ids = rand_ids.shift(@params[:batch_size])
          rand_ids.concat(subset_ids)
          data = samples[subset_ids, true]
          values = y[subset_ids]
          # Calculate gradients for loss function.
          loss_grad = loss_gradient(data, values, weight_vec)
          next if loss_grad.ne(0.0).count.zero?
          # Update weight.
          left_weight_vec, left_weight_sqrsum, left_weight_update =
            update_weight(left_weight_vec, left_weight_sqrsum, left_weight_update,
                          left_weight_gradient(loss_grad, data))
          right_weight_vec, right_weight_sqrsum, right_weight_update =
            update_weight(right_weight_vec, right_weight_sqrsum, right_weight_update,
                          right_weight_gradient(loss_grad, data))
          weight_vec = left_weight_vec - right_weight_vec
        end
        split_weight_vec_bias(weight_vec)
      end

      def loss_gradient(x, y, weight)
        2.0 * (x.dot(weight) - y)
      end

      def left_weight_gradient(loss_grad, data)
        ((@params[:reg_param] + loss_grad).expand_dims(1) * data).mean(0)
      end

      def right_weight_gradient(loss_grad, data)
        ((@params[:reg_param] - loss_grad).expand_dims(1) * data).mean(0)
      end

      def update_weight(weight, sqrsum, update, gr)
        new_sqrsum = @params[:decay] * sqrsum + (1.0 - @params[:decay]) * gr**2
        new_update = (@params[:learning_rate] / ((new_sqrsum + 1.0e-8)**0.5)) * gr
        new_weight = weight - (new_update + @params[:momentum] * update)
        new_weight = 0.5 * (new_weight + new_weight.abs)
        [new_weight, new_sqrsum, new_update]
      end

      def expand_feature(x)
        Numo::NArray.hstack([x, Numo::DFloat.ones([x.shape[0], 1])])
      end

      def split_weight_vec_bias(weight_vec)
        weights = @params[:fit_bias] ? weight_vec[0...-1] : weight_vec
        bias = @params[:fit_bias] ? weight_vec[-1] : 0.0
        [weights, bias]
      end
    end
  end
end
