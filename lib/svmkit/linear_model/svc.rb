require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

module SVMKit
  # This module consists of the classes that implement generalized linear models.
  module LinearModel
    # SVC is a class that implements Support Vector Classifier with the Pegasos algorithm.
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
      # @return [Numo::DFloat] (shape: [n_features])
      attr_reader :weight_vec

      # Return the bias term (a.k.a. intercept) for SVC.
      # @return [Float]
      attr_reader :bias_term

      # Return the random generator for performing random sampling in the Pegasos algorithm.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with Support Vector Machine by the Pegasos algorithm.
      #
      # @param reg_param [Float] The regularization parameter.
      # @param fit_bias [Boolean] The flag indicating whether to fit the bias term.
      # @param bias_scale [Float] The scale of the bias term.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param batch_size [Integer] The size of the mini batches.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(reg_param: 1.0, fit_bias: false, bias_scale: 1.0, max_iter: 100, batch_size: 50, random_seed: nil)
        @params = {}
        @params[:reg_param] = reg_param
        @params[:fit_bias] = fit_bias
        @params[:bias_scale] = bias_scale
        @params[:max_iter] = max_iter
        @params[:batch_size] = batch_size
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @weight_vec = nil
        @bias_term = 0.0
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [SVC] The learned classifier itself.
      def fit(x, y)
        # Generate binary labels
        negative_label = y.to_a.uniq.sort.shift
        bin_y = y.to_a.map { |l| l != negative_label ? 1 : -1 }
        # Expand feature vectors for bias term.
        samples = x
        if @params[:fit_bias]
          samples = Numo::NArray.hstack(
            [samples, Numo::DFloat.ones([x.shape[0], 1]) * @params[:bias_scale]]
          )
        end
        # Initialize some variables.
        n_samples, n_features = samples.shape
        rand_ids = [*0...n_samples].shuffle(random: @rng)
        weight_vec = Numo::DFloat.zeros(n_features)
        # Start optimization.
        @params[:max_iter].times do |t|
          # random sampling
          subset_ids = rand_ids.shift(@params[:batch_size])
          rand_ids.concat(subset_ids)
          target_ids = subset_ids.map { |n| n if weight_vec.dot(samples[n, true]) * bin_y[n] < 1 }.compact
          n_subsamples = target_ids.size
          next if n_subsamples.zero?
          # update the weight vector.
          eta = 1.0 / (@params[:reg_param] * (t + 1))
          mean_vec = Numo::DFloat.zeros(n_features)
          target_ids.each { |n| mean_vec += samples[n, true] * bin_y[n] }
          mean_vec *= eta / n_subsamples
          weight_vec = weight_vec * (1.0 - eta * @params[:reg_param]) + mean_vec
          # scale the weight vector.
          norm = Math.sqrt(weight_vec.dot(weight_vec))
          scaler = (1.0 / @params[:reg_param]**0.5) / (norm + 1.0e-12)
          weight_vec *= [1.0, scaler].min
        end
        # Store the learned model.
        if @params[:fit_bias]
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
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to compute the scores.
      # @return [Numo::DFloat] (shape: [n_samples]) Confidence score per sample.
      def decision_function(x)
        @weight_vec.dot(x.transpose) + @bias_term
      end

      # Predict class labels for samples.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The samples to predict the labels.
      # @return [Numo::Int32] (shape: [n_samples]) Predicted class label per sample.
      def predict(x)
        Numo::Int32.cast(decision_function(x).map { |v| v >= 0 ? 1 : -1 })
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
      # @return [Hash] The marshal data about SVC.
      def marshal_dump
        { params: @params, weight_vec: @weight_vec, bias_term: @bias_term, rng: @rng }
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
    end
  end
end
