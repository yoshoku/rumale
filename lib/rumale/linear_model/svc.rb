# frozen_string_literal: true

require 'rumale/linear_model/base_sgd'
require 'rumale/base/classifier'
require 'rumale/probabilistic_output'

module Rumale
  # This module consists of the classes that implement generalized linear models.
  module LinearModel
    # SVC is a class that implements Support Vector Classifier
    # with stochastic gradient descent optimization.
    # For multiclass classification problem, it uses one-vs-the-rest strategy.
    #
    # Rumale::SVM provides linear support vector classifier based on LIBLINEAR.
    # If you prefer execution speed, you should use Rumale::SVM::LinearSVC.
    # https://github.com/yoshoku/rumale-svm
    #
    # @example
    #   estimator =
    #     Rumale::LinearModel::SVC.new(reg_param: 1.0, max_iter: 200, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Shalev-Shwartz and Y. Singer, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Proc. ICML'07, pp. 807--814, 2007.
    # - Y. Tsuruoka, J. Tsujii, and S. Ananiadou, "Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty," Proc. ACL'09, pp. 477--485, 2009.
    # - L. Bottou, "Large-Scale Machine Learning with Stochastic Gradient Descent," Proc. COMPSTAT'10, pp. 177--186, 2010.
    class SVC < BaseSGD
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
      # @param learning_rate [Float] The initial value of learning rate.
      #   The learning rate decreases as the iteration proceeds according to the equation: learning_rate / (1 + decay * t).
      # @param decay [Float] The smoothing parameter for decreasing learning rate as the iteration proceeds.
      #   If nil is given, the decay sets to 'reg_param * learning_rate'.
      # @param momentum [Float] The momentum factor.
      # @param penalty [String] The regularization type to be used ('l1', 'l2', and 'elasticnet').
      # @param l1_ratio [Float] The elastic-net type regularization mixing parameter.
      #   If penalty set to 'l2' or 'l1', this parameter is ignored.
      #   If l1_ratio = 1, the regularization is similar to Lasso.
      #   If l1_ratio = 0, the regularization is similar to Ridge.
      #   If 0 < l1_ratio < 1, the regularization is a combination of L1 and L2.
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      # @param batch_size [Integer] The size of the mini batches.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      # @param probability [Boolean] The flag indicating whether to perform probability estimation.
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(learning_rate: 0.01, decay: nil, momentum: 0.9,
                     penalty: 'l2', reg_param: 1.0, l1_ratio: 0.5,
                     fit_bias: true, bias_scale: 1.0,
                     max_iter: 200, batch_size: 50, tol: 1e-4,
                     probability: false,
                     n_jobs: nil, verbose: false, random_seed: nil)
        check_params_numeric(learning_rate: learning_rate, momentum: momentum,
                             reg_param: reg_param, l1_ratio: l1_ratio, bias_scale: bias_scale,
                             max_iter: max_iter, batch_size: batch_size, tol: tol)
        check_params_boolean(fit_bias: fit_bias, verbose: verbose, probability: probability)
        check_params_string(penalty: penalty)
        check_params_numeric_or_nil(decay: decay, n_jobs: n_jobs, random_seed: random_seed)
        check_params_positive(learning_rate: learning_rate, reg_param: reg_param,
                              bias_scale: bias_scale, max_iter: max_iter, batch_size: batch_size)
        super()
        @params.merge!(method(:initialize).parameters.map { |_t, arg| [arg, binding.local_variable_get(arg)] }.to_h)
        @params[:decay] ||= @params[:reg_param] * @params[:learning_rate]
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @penalty_type = @params[:penalty]
        @loss_func = LinearModel::Loss::HingeLoss.new
        @weight_vec = nil
        @bias_term = nil
        @classes = nil
        @prob_param = nil
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [SVC] The learned classifier itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]

        if multiclass_problem?
          n_classes = @classes.size
          n_features = x.shape[1]
          # initialize model.
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @bias_term = Numo::DFloat.zeros(n_classes)
          @prob_param = Numo::DFloat.zeros(n_classes, 2)
          # fit model.
          models = if enable_parallel?
                     # :nocov:
                     parallel_map(n_classes) do |n|
                       bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
                       partial_fit(x, bin_y)
                     end
                     # :nocov:
                   else
                     Array.new(n_classes) do |n|
                       bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
                       partial_fit(x, bin_y)
                     end
                   end
          # store model.
          models.each_with_index { |model, n| @weight_vec[n, true], @bias_term[n], @prob_param[n, true] = model }
        else
          negative_label = @classes[0]
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec, @bias_term, @prob_param = partial_fit(x, bin_y)
        end

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        x = check_convert_sample_array(x)
        x.dot(@weight_vec.transpose) + @bias_term
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        x = check_convert_sample_array(x)

        n_samples = x.shape[0]
        predicted = if multiclass_problem?
                      decision_values = decision_function(x)
                      if enable_parallel?
                        parallel_map(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                      else
                        Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                      end
                    else
                      decision_values = decision_function(x).ge(0.0).to_a
                      Array.new(n_samples) { |n| @classes[decision_values[n]] }
                    end
        Numo::Int32.asarray(predicted)
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = check_convert_sample_array(x)

        if multiclass_problem?
          probs = 1.0 / (Numo::NMath.exp(@prob_param[true, 0] * decision_function(x) + @prob_param[true, 1]) + 1.0)
          (probs.transpose / probs.sum(axis: 1)).transpose.dup
        else
          n_samples, = x.shape
          probs = Numo::DFloat.zeros(n_samples, 2)
          probs[true, 1] = 1.0 / (Numo::NMath.exp(@prob_param[0] * decision_function(x) + @prob_param[1]) + 1.0)
          probs[true, 0] = 1.0 - probs[true, 1]
          probs
        end
      end

      private

      def partial_fit(x, bin_y)
        w, b = super
        p = if @params[:probability]
              Rumale::ProbabilisticOutput.fit_sigmoid(x.dot(w.transpose) + b, bin_y)
            else
              Numo::DFloat[1, 0]
            end
        [w, b, p]
      end

      def multiclass_problem?
        @classes.size > 2
      end
    end
  end
end
