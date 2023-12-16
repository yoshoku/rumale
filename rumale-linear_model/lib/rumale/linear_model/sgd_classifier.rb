# frozen_string_literal: true

require 'rumale/base/classifier'
require 'rumale/probabilistic_output'
require 'rumale/validation'

require_relative 'sgd_estimator'

module Rumale
  module LinearModel
    # SGDClassifier is a class that implements linear classifier with stochastic gradient descent optimization.
    #
    # @example
    #   require 'rumale/linear_model/sgd_classifier'
    #
    #   estimator =
    #     Rumale::LinearModel::SGDClassifier.new(loss: 'hinge', reg_param: 1.0, max_iter: 1000, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - Shalev-Shwartz, S., and Singer, Y., "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Proc. ICML'07, pp. 807--814, 2007.
    # - Tsuruoka, Y., Tsujii, J., and Ananiadou, S., "Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty," Proc. ACL'09, pp. 477--485, 2009.
    # - Bottou, L., "Large-Scale Machine Learning with Stochastic Gradient Descent," Proc. COMPSTAT'10, pp. 177--186, 2010.
    class SGDClassifier < Rumale::LinearModel::SGDEstimator # rubocop:disable Metrics/ClassLength
      include Rumale::Base::Classifier

      # Return the class labels.
      # @return [Numo::Int32] (shape: [n_classes])
      attr_reader :classes

      # Return the random generator for performing random sampling.
      # @return [Random]
      attr_reader :rng

      # Create a new linear classifier with stochastic gradient descent optimization.
      #
      # @param loss [String] The loss function to be used ('hinge' and 'log_loss').
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
      # @param n_jobs [Integer] The number of jobs for running the fit and predict methods in parallel.
      #   If nil is given, the methods do not execute in parallel.
      #   If zero or less is given, it becomes equal to the number of processors.
      #   This parameter is ignored if the Parallel gem is not loaded.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(loss: 'hinge', learning_rate: 0.01, decay: nil, momentum: 0.9,
                     penalty: 'l2', reg_param: 1.0, l1_ratio: 0.5,
                     fit_bias: true, bias_scale: 1.0,
                     max_iter: 1000, batch_size: 50, tol: 1e-4,
                     n_jobs: nil, verbose: false, random_seed: nil)
        super()
        @params.merge!(
          loss: loss,
          learning_rate: learning_rate,
          decay: decay,
          momentum: momentum,
          penalty: penalty,
          reg_param: reg_param,
          l1_ratio: l1_ratio,
          fit_bias: fit_bias,
          bias_scale: bias_scale,
          max_iter: max_iter,
          batch_size: batch_size,
          tol: tol,
          n_jobs: n_jobs,
          verbose: verbose,
          random_seed: random_seed
        )
        @params[:decay] ||= @params[:reg_param] * @params[:learning_rate]
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @penalty_type = @params[:penalty]
        @loss_func = case @params[:loss]
                     when Rumale::LinearModel::Loss::HingeLoss::NAME
                       Rumale::LinearModel::Loss::HingeLoss.new
                     when Rumale::LinearModel::Loss::LogLoss::NAME
                       Rumale::LinearModel::Loss::LogLoss.new
                     else
                       raise ArgumentError, "given loss '#{loss}' is not supported."
                     end
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [SGDClassifier] The learned classifier itself.
      def fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_label_array(y)
        Rumale::Validation.check_sample_size(x, y)

        @classes = Numo::Int32[*y.to_a.uniq.sort]

        send(:"fit_#{@loss_func.name}", x, y)

        self
      end

      # Perform 1-epoch of stochastic gradient descent optimization with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The binary labels to be used for fitting the model.
      # @return [SGDClassifier] The learned classifier itself.
      def partial_fit(x, y)
        x = Rumale::Validation.check_convert_sample_array(x)
        y = Rumale::Validation.check_convert_label_array(y)
        Rumale::Validation.check_sample_size(x, y)

        n_features = x.shape[1]
        n_features += 1 if fit_bias?
        need_init = @weight.nil? || @weight.shape[0] != n_features

        @classes = Numo::Int32[*y.to_a.uniq.sort] if need_init
        negative_label = @classes[0]
        bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1

        @weight_vec, @bias_term = partial_fit_(x, bin_y, max_iter: 1, init: need_init)
        if @loss_func.name == Rumale::LinearModel::Loss::HingeLoss::NAME
          @prob_param = Rumale::ProbabilisticOutput.fit_sigmoid(x.dot(@weight_vec.transpose) + @bias_term, bin_y)
        end

        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Confidence score per sample.
      def decision_function(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        x.dot(@weight_vec.transpose) + @bias_term
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        send(:"predict_#{@loss_func.name}", x)
      end

      # Predict probability for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [Numo::DFloat] (shape: [n_samples, n_classes]) Predicted probability of each class per sample.
      def predict_proba(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        send(:"predict_proba_#{@loss_func.name}", x)
      end

      private

      def fit_hinge(x, y)
        if multiclass_problem?
          n_classes = @classes.size
          n_features = x.shape[1]
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @bias_term = Numo::DFloat.zeros(n_classes)
          @prob_param = Numo::DFloat.zeros(n_classes, 2)
          models = if enable_parallel?
                     parallel_map(n_classes) do |n|
                       bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
                       w, b = partial_fit_(x, bin_y)
                       prb = Rumale::ProbabilisticOutput.fit_sigmoid(x.dot(w.transpose) + b, bin_y)
                       [w, b, prb]
                     end
                   else
                     Array.new(n_classes) do |n|
                       bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
                       w, b = partial_fit_(x, bin_y)
                       prb = Rumale::ProbabilisticOutput.fit_sigmoid(x.dot(w.transpose) + b, bin_y)
                       [w, b, prb]
                     end
                   end
          # store model.
          models.each_with_index { |model, n| @weight_vec[n, true], @bias_term[n], @prob_param[n, true] = model }
        else
          negative_label = @classes[0]
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec, @bias_term = partial_fit_(x, bin_y)
          @prob_param = Rumale::ProbabilisticOutput.fit_sigmoid(x.dot(@weight_vec.transpose) + @bias_term, bin_y)
        end
      end

      def fit_log_loss(x, y)
        if multiclass_problem?
          n_classes = @classes.size
          n_features = x.shape[1]
          @weight_vec = Numo::DFloat.zeros(n_classes, n_features)
          @bias_term = Numo::DFloat.zeros(n_classes)
          if enable_parallel?
            models = parallel_map(n_classes) do |n|
              bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
              partial_fit_(x, bin_y)
            end
            n_classes.times { |n| @weight_vec[n, true], @bias_term[n] = models[n] }
          else
            n_classes.times do |n|
              bin_y = Numo::Int32.cast(y.eq(@classes[n])) * 2 - 1
              @weight_vec[n, true], @bias_term[n] = partial_fit_(x, bin_y)
            end
          end
        else
          negative_label = @classes[0]
          bin_y = Numo::Int32.cast(y.ne(negative_label)) * 2 - 1
          @weight_vec, @bias_term = partial_fit_(x, bin_y)
        end
      end

      def predict_proba_hinge(x)
        if multiclass_problem?
          probs = 1.0 / (Numo::NMath.exp(@prob_param[true, 0] * decision_function(x) + @prob_param[true, 1]) + 1.0)
          (probs.transpose / probs.sum(axis: 1)).transpose.dup
        else
          n_samples = x.shape[0]
          probs = Numo::DFloat.zeros(n_samples, 2)
          probs[true, 1] = 1.0 / (Numo::NMath.exp(@prob_param[0] * decision_function(x) + @prob_param[1]) + 1.0)
          probs[true, 0] = 1.0 - probs[true, 1]
          probs
        end
      end

      def predict_proba_log_loss(x)
        proba = 1.0 / (Numo::NMath.exp(-decision_function(x)) + 1.0)
        return (proba.transpose / proba.sum(axis: 1)).transpose.dup if multiclass_problem?

        n_samples = x.shape[0]
        probs = Numo::DFloat.zeros(n_samples, 2)
        probs[true, 1] = proba
        probs[true, 0] = 1.0 - proba
        probs
      end

      def predict_hinge(x)
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

      def predict_log_loss(x)
        n_samples = x.shape[0]
        decision_values = predict_proba_log_loss(x)
        predicted = if enable_parallel?
                      parallel_map(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                    else
                      Array.new(n_samples) { |n| @classes[decision_values[n, true].max_index] }
                    end
        Numo::Int32.asarray(predicted)
      end

      def multiclass_problem?
        @classes.size > 2
      end
    end
  end
end
