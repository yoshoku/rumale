require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

module SVMKit
  # This module consists of the classes that implemnt polynomial models.
  module PolynomialModel
    # FactorizationMachineClassifier is a class that
    # implements Fatorization Machine for binary classification
    # with (mini-batch) stochastic gradient descent optimization.
    # Note that this implementation uses hinge loss for the loss function.
    #
    # @example
    #   estimator =
    #     SVMKit::PolynomialModel::FactorizationMachineClassifier.new(
    #      n_factors: 10, reg_param_bias: 0.001, reg_param_weight: 0.001, reg_param_factor: 0.001,
    #      max_iter: 5000, batch_size: 50, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # *Reference*
    # - S. Rendle, "Factorization Machines with libFM," ACM Transactions on Intelligent Systems and Technology, vol. 3 (3), pp. 57:1--57:22, 2012.
    # - S. Rendle, "Factorization Machines," Proceedings of the 10th IEEE International Conference on Data Mining (ICDM'10), pp. 995--1000, 2010.
    class FactorizationMachineClassifier
      include Base::BaseEstimator
      include Base::Classifier

      # Return the factor matrix for Factorization Machine.
      # @return [Numo::DFloat] (shape: [n_factors, n_features])
      attr_reader :factor_mat

      # Return the weight vector for Factorization Machine.
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :weight_vec

      # Return the bias term for Factoriazation Machine.
      # @return [Float]
      attr_reader :bias_term

      # Return the random generator for transformation.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with Support Vector Machine by the Pegasos algorithm.
      #
      # @param n_factors [Integer] The maximum number of iterations.
      # @param reg_param_bias [Float] The regularization parameter for bias term.
      # @param reg_param_weight [Float] The regularization parameter for weight vector.
      # @param reg_param_factor [Float] The regularization parameter for factor matrix.
      # @param init_std [Float] The standard deviation of normal random number for initialization of factor matrix.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_factors: 2, reg_param_bias: 1.0, reg_param_weight: 1.0, reg_param_factor: 1.0,
                     init_std: 0.1, max_iter: 1000, batch_size: 10, random_seed: nil)
        @params = {}
        @params[:n_factors] = n_factors
        @params[:reg_param_bias] = reg_param_bias
        @params[:reg_param_weight] = reg_param_weight
        @params[:reg_param_factor] = reg_param_factor
        @params[:init_std] = init_std
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @factor_mat = nil
        @weight_vec = nil
        @bias_term = 0.0
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [FactorizationMachineClassifier] The learned classifier itself.
      def fit(x, y)
        # Generate binary labels.
        negative_label = y.to_a.uniq.sort.shift
        bin_y = y.map { |l| l != negative_label ? 1.0 : -1.0 }
        # Initialize some variables.
        n_samples, n_features = x.shape
        rand_ids = [*0...n_samples].shuffle(random: @rng)
        @factor_mat = rand_normal([@params[:n_factors], n_features], 0, @params[:init_std])
        @weight_vec = Numo::DFloat.zeros(n_features)
        @bias_term = 0.0
        # Start optimization.
        @params[:max_iter].times do |t|
          # Random sampling.
          subset_ids = rand_ids.shift(@params[:batch_size])
          rand_ids.concat(subset_ids)
          data = x[subset_ids, true]
          label = bin_y[subset_ids]
          # Calculate gradients for loss function.
          loss_grad = loss_gradient(data, label)
          next if loss_grad.ne(0.0).count.zero?
          # Update each parameter.
          @bias_term -= learning_rate(@params[:reg_param_bias], t) * bias_gradient(loss_grad)
          @weight_vec -= learning_rate(@params[:reg_param_weight], t) * weight_gradient(loss_grad, data)
          @params[:n_factors].times do |n|
            @factor_mat[n, true] -= learning_rate(@params[:reg_param_factor], t) *
                                    factor_gradient(loss_grad, data, @factor_mat[n, true])
          end
        end
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples]) Confidence score per sample.
      def decision_function(x)
        linear_term = @bias_term + x.dot(@weight_vec)
        factor_term = 0.5 * (@factor_mat.dot(x.transpose)**2 - (@factor_mat**2).dot(x.transpose**2)).sum
        linear_term + factor_term
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        Numo::Int32.cast(decision_function(x).map { |v| v >= 0.0 ? 1 : -1 })
      end

      # Claculate the mean accuracy of the given testing data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) Testing data.
      # @param y [Numo::Int32] (shape: [n_samples]) True labels for testing data.
      # @return [Float] Mean accuracy
      def score(x, y)
        p = predict(x)
        n_hits = (y.to_a.map.with_index { |l, n| l == p[n] ? 1 : 0 }).inject(:+)
        n_hits / y.size.to_f
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about FactorizationMachineClassifier
      def marshal_dump
        { params: @params, factor_mat: @factor_mat, weight_vec: @weight_vec, bias_term: @bias_term, rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @factor_mat = obj[:factor_mat]
        @weight_vec = obj[:weight_vec]
        @bias_term = obj[:bias_term]
        @rng = obj[:rng]
        nil
      end

      private

      def loss_gradient(x, y)
        evaluated = y * decision_function(x)
        gradient = Numo::DFloat.zeros(evaluated.size)
        gradient[evaluated < 1.0] = -y[evaluated < 1.0]
        gradient
      end

      def learning_rate(reg_param, iter)
        1.0 / (reg_param * (iter + 1))
      end

      def bias_gradient(loss_grad)
        loss_grad.mean + @params[:reg_param_bias] * @bias_term
      end

      def weight_gradient(loss_grad, data)
        (loss_grad.expand_dims(1) * data).mean(0) + @params[:reg_param_weight] * @weight_vec
      end

      def factor_gradient(loss_grad, data, factor)
        reg_term = @params[:reg_param_factor] * factor
        (loss_grad.expand_dims(1) * (data * data.dot(factor).expand_dims(1) - factor * (data**2))).mean(0) + reg_term
      end

      def rand_uniform(shape)
        Numo::DFloat[*Array.new(shape.inject(&:*)) { @rng.rand }].reshape(*shape)
      end

      def rand_normal(shape, mu, sigma)
        a = rand_uniform(shape)
        b = rand_uniform(shape)
        mu + sigma * (Numo::NMath.sqrt(-2.0 * Numo::NMath.log(a)) * Numo::NMath.sin(2.0 * Math::PI * b))
      end
    end
  end
end
