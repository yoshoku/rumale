require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

module SVMKit
  # This module consists of the classes that implement generalized linear models.
  module LinearModel
    # LogisticRegression is a class that implements Logistic Regression
    # with stochastic gradient descent (SGD) optimization.
    # Note that the class performs as a binary classifier.
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

      # @!visibility private
      DEFAULT_PARAMS = {
        reg_param: 1.0,
        fit_bias: false,
        bias_scale: 1.0,
        max_iter: 100,
        batch_size: 50,
        random_seed: nil
      }.freeze

      # Return the weight vector for Logistic Regression.
      # @return [NMatrix] (shape: [1, n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept) for Logistic Regression.
      # @return [Float]
      attr_reader :bias_term

      # Return the random generator for transformation.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with Logisitc Regression by the SGD optimization.
      #
      # @overload new(reg_param: 1.0, max_iter: 100, batch_size: 50, random_seed: 1) -> LogisiticRegression
      #
      # @param reg_param   [Float] (defaults to: 1.0) The regularization parameter.
      # @param fit_bias    [Boolean] (defaults to: false) The flag indicating whether to fit the bias term.
      # @param bias_scale  [Float] (defaults to: 1.0) The scale of the bias term.
      #   If fit_bias is true, the feature vector v becoms [v; bias_scale].
      # @param max_iter    [Integer] (defaults to: 100) The maximum number of iterations.
      # @param batch_size  [Integer] (defaults to: 50) The size of the mini batches.
      # @param random_seed [Integer] (defaults to: nil) The seed value using to initialize the random generator.
      def initialize(params = {})
        self.params = DEFAULT_PARAMS.merge(Hash[params.map { |k, v| [k.to_sym, v] }])
        self.params[:random_seed] ||= srand
        @weight_vec = nil
        @bias_term = 0.0
        @rng = Random.new(self.params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [NMatrix] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [NMatrix] (shape: [1, n_samples]) The categorical variables (e.g. labels)
      #   to be used for fitting the model.
      # @return [LogisticRegression] The learned classifier itself.
      def fit(x, y)
        # Generate binary labels.
        negative_label = y.uniq.sort.shift
        bin_y = y.to_flat_a.map { |l| l != negative_label ? 1 : 0 }
        # Expand feature vectors for bias term.
        samples = x
        samples = samples.hconcat(NMatrix.ones([x.shape[0], 1]) * params[:bias_scale]) if params[:fit_bias]
        # Initialize some variables.
        n_samples, n_features = samples.shape
        rand_ids = [*0..n_samples - 1].shuffle(random: @rng)
        weight_vec = NMatrix.zeros([1, n_features])
        # Start optimization.
        params[:max_iter].times do |t|
          # random sampling
          subset_ids = rand_ids.shift(params[:batch_size])
          rand_ids.concat(subset_ids)
          # update the weight vector.
          eta = 1.0 / (params[:reg_param] * (t + 1))
          mean_vec = NMatrix.zeros([1, n_features])
          subset_ids.each do |n|
            z = weight_vec.dot(samples.row(n).transpose)[0]
            coef = bin_y[n] / (1.0 + Math.exp(bin_y[n] * z))
            mean_vec += samples.row(n) * coef
          end
          mean_vec *= eta / params[:batch_size]
          weight_vec = weight_vec * (1.0 - eta * params[:reg_param]) + mean_vec
          # scale the weight vector.
          scaler = (1.0 / params[:reg_param]**0.5) / weight_vec.norm2
          weight_vec *= [1.0, scaler].min
        end
        # Store the learned model.
        if params[:fit_bias]
          @weight_vec = weight_vec[0...n_features - 1]
          @bias_term = weight_vec[n_features - 1]
        else
          @weight_vec = weight_vec[0...n_features]
          @bias_term = 0.0
        end
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [NMatrix] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [NMatrix] (shape: [1, n_samples]) Confidence score per sample.
      def decision_function(x)
        w = ((@weight_vec.dot(x.transpose) + @bias_term) * -1.0).exp + 1.0
        w.map { |v| 1.0 / v }
      end

      # Predict class labels for samples.
      #
      # @param x [NMatrix] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [NMatrix] (shape: [1, n_samples]) Predicted class label per sample.
      def predict(x)
        decision_function(x).map { |v| v >= 0.5 ? 1 : -1 }
      end

      # Predict probability for samples.
      #
      # @param x [NMatrix] (shape: [n_samples, n_features]) The samples to predict the probailities.
      # @return [NMatrix] (shape: [1, n_samples]) Predicted probability per sample.
      def predict_proba(x)
        decision_function(x)
      end

      # Claculate the mean accuracy of the given testing data.
      #
      # @param x [NMatrix] (shape: [n_samples, n_features]) Testing data.
      # @param y [NMatrix] (shape: [1, n_samples]) True labels for testing data.
      # @return [Float] Mean accuracy
      def score(x, y)
        p = predict(x)
        n_hits = (y.to_flat_a.map.with_index { |l, n| l == p[n] ? 1 : 0 }).inject(:+)
        n_hits / y.size.to_f
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about LogisticRegression.
      def marshal_dump
        { params: params, weight_vec: Utils.dump_nmatrix(@weight_vec), bias_term: @bias_term, rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        self.params = obj[:params]
        @weight_vec = Utils.restore_nmatrix(obj[:weight_vec])
        @bias_term = obj[:bias_term]
        @rng = obj[:rng]
        nil
      end
    end
  end
end
