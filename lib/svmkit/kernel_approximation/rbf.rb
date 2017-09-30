require 'svmkit/base/base_estimator'
require 'svmkit/base/transformer'

module SVMKit
  # Module for kernel approximation algorithms.
  module KernelApproximation
    # Class for RBF kernel feature mapping.
    #
    #   transformer = SVMKit::KernelApproximation::RBF.new(gamma: 1.0, n_coponents: 128, random_seed: 1)
    #   new_training_samples = transformer.fit_transform(training_samples)
    #   new_testing_samples = transformer.transform(testing_samples)
    #
    # * *Refernce*:
    #   - A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines," Proc. NIPS'07, pp.1177--1184, 2007.
    class RBF
      include Base::BaseEstimator
      include Base::Transformer

      DEFAULT_PARAMS = { # :nodoc:
        gamma: 1.0,
        n_components: 128,
        random_seed: nil
      }.freeze

      # The random matrix for transformation.
      attr_reader :random_mat # :nodoc:

      # The random vector for transformation.
      attr_reader :random_vec # :nodoc:

      # The random generator for transformation.
      attr_reader :rng # :nodoc:

      # Creates a new transformer for mapping to RBF kernel feature space.
      #
      # call-seq:
      #   new(gamma: 1.0, n_components: 128, random_seed: 1) -> RBF
      #
      # * *Arguments* :
      #   - +:gamma+ (Float) (defaults to: 1.0) -- The parameter of RBF kernel: exp(-gamma * x^2)
      #   - +:n_components+ (Integer) (defaults to: 128) -- The number of dimensions of the RBF kernel feature space.
      #   - +:random_seed+ (Integer) (defaults to: nil) -- The seed value using to initialize the random generator.
      def initialize(params = {})
        self.params = DEFAULT_PARAMS.merge(Hash[params.map { |k, v| [k.to_sym, v] }])
        self.params[:random_seed] ||= srand
        @rng = Random.new(self.params[:random_seed])
        @random_mat = nil
        @random_vec = nil
      end

      # Fit the model with given training data.
      #
      # call-seq:
      #   fit(x) -> RBF
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The training data to be used for fitting the model. This method uses only the number of features of the data.
      # * *Returns* :
      #   - The learned transformer itself.
      def fit(x, _y = nil)
        n_features = x.shape[1]
        params[:n_components] = 2 * n_features if params[:n_components] <= 0
        @random_mat = rand_normal([n_features, params[:n_components]]) * (2.0 * params[:gamma])**0.5
        n_half_components = params[:n_components] / 2
        @random_vec = NMatrix.zeros([1, params[:n_components] - n_half_components]).hconcat(
          NMatrix.ones([1, n_half_components]) * (0.5 * Math::PI)
        )
        #@random_vec = rand_uniform([1, self.params[:n_components]]) * (2.0 * Math::PI)
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # call-seq:
      #   fit_transform(x) -> NMatrix
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The training data to be used for fitting the model.
      # * *Returns* :
      #   - The transformed data (NMatrix, shape: [n_samples, n_components]).
      def fit_transform(x, _y = nil)
        fit(x).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # call-seq:
      #   transform(x) -> NMatrix
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The data to be transformed with the learned model.
      # * *Returns* :
      #   - The transformed data (NMatrix, shape: [n_samples, n_components]).
      def transform(x)
        n_samples, = x.shape
        projection = x.dot(@random_mat) + @random_vec.repeat(n_samples, 0)
        projection.sin * ((2.0 / params[:n_components])**0.5)
      end

      # Serializes object through Marshal#dump.
      def marshal_dump # :nodoc:
        { params: params,
          random_mat: Utils.dump_nmatrix(@random_mat),
          random_vec: Utils.dump_nmatrix(@random_vec),
          rng: @rng }
      end

      # Deserialize object through Marshal#load.
      def marshal_load(obj) # :nodoc:
        self.params = obj[:params]
        @random_mat = Utils.restore_nmatrix(obj[:random_mat])
        @random_vec = Utils.restore_nmatrix(obj[:random_vec])
        @rng = obj[:rng]
        nil
      end

      protected

      # Generate the uniform random matrix with the given shape.
      def rand_uniform(shape) # :nodoc:
        rnd_vals = Array.new(NMatrix.size(shape)) { @rng.rand }
        NMatrix.new(shape, rnd_vals, dtype: :float64, stype: :dense)
      end

      # Generate the normal random matrix with the given shape, mean, and standard deviation.
      def rand_normal(shape, mu = 0.0, sigma = 1.0) # :nodoc:
        a = rand_uniform(shape)
        b = rand_uniform(shape)
        ((a.log * -2.0).sqrt * (b * 2.0 * Math::PI).sin) * sigma + mu
      end
    end
  end
end
