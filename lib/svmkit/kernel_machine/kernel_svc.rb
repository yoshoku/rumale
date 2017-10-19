require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

module SVMKit
  # This module consists of the classes that implement kernel method-based estimator.
  module KernelMachine
    # KernelSVC is a class that implements (Nonlinear) Kernel Support Vector Classifier with the Pegasos algorithm.
    #
    # @example
    #   training_kernel_matrix = SVMKit::PairwiseMetric::rbf_kernel(training_samples)
    #   estimator =
    #     SVMKit::KernelMachine::KernelSVC.new(reg_param: 1.0, max_iter: 1000, random_seed: 1)
    #   estimator.fit(training_kernel_matrix, traininig_labels)
    #   testing_kernel_matrix = SVMKit::PairwiseMetric::rbf_kernel(testing_samples, training_samples)
    #   results = estimator.predict(testing_kernel_matrix)
    #
    # *Reference*
    # 1. S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Mathematical Programming, vol. 127 (1), pp. 3--30, 2011.
    class KernelSVC
      include Base::BaseEstimator
      include Base::Classifier

      # @!visibility private
      DEFAULT_PARAMS = {
        reg_param: 1.0,
        max_iter: 1000,
        random_seed: nil
      }.freeze

      # Return the weight vector for Kernel SVC.
      # @return [NMatrix] (shape: [1, n_trainig_sample])
      attr_reader :weight_vec

      # Return the random generator for performing random sampling in the Pegasos algorithm.
      # @return [Random]
      attr_reader :rng

      # Create a new classifier with Kernel Support Vector Machine by the Pegasos algorithm.
      #
      # @overload new(reg_param: 1.0, max_iter: 1000, random_seed: 1) -> KernelSVC
      #
      # @param params [Hash] The parameters for Kernel SVC.
      # @option params [Float]   :reg_param (1.0) The regularization parameter.
      # @option params [Integer] :max_iter (1000) The maximum number of iterations.
      # @option params [Integer] :random_seed (nil) The seed value using to initialize the random generator.
      def initialize(params = {})
        self.params = DEFAULT_PARAMS.merge(Hash[params.map { |k, v| [k.to_sym, v] }])
        self.params[:random_seed] ||= srand
        @weight_vec = nil
        @rng = Random.new(self.params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [NMatrix] (shape: [n_training_samples, n_training_samples])
      #   The kernel matrix of the training data to be used for fitting the model.
      # @param y [NMatrix] (shape: [1, n_training_samples]) The labels to be used for fitting the model.
      # @return [KernelSVC] The learned classifier itself.
      def fit(x, y)
        # Generate binary labels
        negative_label = y.uniq.sort.shift
        bin_y = y.to_flat_a.map { |l| l != negative_label ? 1 : -1 }
        # Initialize some variables.
        n_training_samples = x.shape[0]
        rand_ids = []
        weight_vec = NMatrix.zeros([1, n_training_samples])
        # Start optimization.
        params[:max_iter].times do |t|
          # random sampling
          rand_ids = [*0...n_training_samples].shuffle(random: @rng) if rand_ids.empty?
          target_id = rand_ids.shift
          # update the weight vector
          func = (weight_vec * bin_y[target_id]).dot(x.row(target_id).transpose).to_f
          func *= bin_y[target_id] / (params[:reg_param] * (t + 1))
          weight_vec[target_id] += 1.0 if func < 1.0
        end
        # Store the learned model.
        @weight_vec = weight_vec * NMatrix.new([1, n_training_samples], bin_y)
        self
      end

      # Calculate confidence scores for samples.
      #
      # @param x [NMatrix] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to compute the scores.
      # @return [NMatrix] (shape: [1, n_testing_samples]) Confidence score per sample.
      def decision_function(x)
        @weight_vec.dot(x.transpose)
      end

      # Predict class labels for samples.
      #
      # @param x [NMatrix] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples to predict the labels.
      # @return [NMatrix] (shape: [1, n_testing_samples]) Predicted class label per sample.
      def predict(x)
        decision_function(x).map { |v| v >= 0 ? 1 : -1 }
      end

      # Claculate the mean accuracy of the given testing data.
      #
      # @param x [NMatrix] (shape: [n_testing_samples, n_training_samples])
      #     The kernel matrix between testing samples and training samples.
      # @param y [NMatrix] (shape: [1, n_testing_samples]) True labels for testing data.
      # @return [Float] Mean accuracy
      def score(x, y)
        p = predict(x)
        n_hits = (y.to_flat_a.map.with_index { |l, n| l == p[n] ? 1 : 0 }).inject(:+)
        n_hits / y.size.to_f
      end

      # Dump marshal data.
      # @return [Hash] The marshal data about KernelSVC.
      def marshal_dump
        { params: params, weight_vec: Utils.dump_nmatrix(@weight_vec), rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        self.params = obj[:params]
        @weight_vec = Utils.restore_nmatrix(obj[:weight_vec])
        @rng = obj[:rng]
        nil
      end
    end
  end
end
