# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'
require 'rumale/utils'
require 'rumale/pairwise_metric'
require 'rumale/decomposition/pca'

module Rumale
  module Manifold
    # MDS is a class that implements Metric Multidimensional Scaling (MDS)
    # with Scaling by MAjorizing a COmplicated Function (SMACOF) algorithm.
    #
    # @example
    #   mds = Rumale::Manifold::MDS.new(init: 'pca', max_iter: 500, random_seed: 1)
    #   representations = mds.fit_transform(samples)
    #
    # *Reference*
    # - P J. F. Groenen and M. van de Velden, "Multidimensional Scaling by Majorization: A Review," J. of Statistical Software, Vol. 73 (8), 2016.
    class MDS
      include Base::BaseEstimator
      include Base::Transformer

      # Return the data in representation space.
      # @return [Numo::DFloat] (shape: [n_samples, n_components])
      attr_reader :embedding

      # Return the stress function value after optimization.
      # @return [Float]
      attr_reader :stress

      # Return the number of iterations run for optimization
      # @return [Integer]
      attr_reader :n_iter

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer with MDS.
      #
      # @param n_components [Integer] The number of dimensions on representation space.
      # @param metric [String] The metric to calculate the distances in original space.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance in original space.
      #   If metric is 'precomputed', the fit and fit_transform methods expect to be given a distance matrix.
      # @param init [String] The init is a method to initialize the representaion space.
      #   If init is 'random', the representaion space is initialized with normal random variables.
      #   If init is 'pca', the result of principal component analysis as the initial value of the representation space.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of stress value for terminating optimization.
      #   If tol is nil,  it does not use stress value as a criterion for terminating the optimization.
      # @param verbose [Boolean] The flag indicating whether to output stress value during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_components: 2, metric: 'euclidean', init: 'random',
                     max_iter: 300, tol: nil, verbose: false, random_seed: nil)
        check_params_integer(n_components: n_components, max_iter: max_iter)
        check_params_string(metric: metric, init: init)
        check_params_boolean(verbose: verbose)
        check_params_type_or_nil(Float, tol: tol)
        check_params_type_or_nil(Integer, random_seed: random_seed)
        check_params_positive(n_components: n_components, max_iter: max_iter)
        @params = {}
        @params[:n_components] = n_components
        @params[:max_iter] = max_iter
        @params[:tol] = tol
        @params[:metric] = metric
        @params[:init] = init
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @embedding = nil
        @stress = nil
        @n_iter = nil
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> MDS
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [MDS] The learned transformer itself.
      def fit(x, _not_used = nil)
        x = check_convert_sample_array(x)
        raise ArgumentError, 'Expect the input distance matrix to be square.' if @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
        # initialize some varibales.
        n_samples = x.shape[0]
        hi_distance_mat = @params[:metric] == 'precomputed' ? x : Rumale::PairwiseMetric.euclidean_distance(x)
        @embedding = init_embedding(x)
        lo_distance_mat = Rumale::PairwiseMetric.euclidean_distance(@embedding)
        @stress = calc_stress(hi_distance_mat, lo_distance_mat)
        @n_iter = 0
        # perform optimization.
        @params[:max_iter].times do |t|
          # guttman tarnsform.
          ratio = hi_distance_mat / lo_distance_mat
          ratio[ratio.diag_indices] = 0.0
          ratio[lo_distance_mat.eq(0)] = 0.0
          tmp_mat = -ratio
          tmp_mat[tmp_mat.diag_indices] += ratio.sum(axis: 1)
          @embedding = 1.fdiv(n_samples) * tmp_mat.dot(@embedding)
          # check convergence.
          new_stress = calc_stress(hi_distance_mat, lo_distance_mat)
          if terminate?(@stress, new_stress)
            @stress = new_stress
            break
          end
          # next step.
          @n_iter = t + 1
          @stress = new_stress
          lo_distance_mat = Rumale::PairwiseMetric.euclidean_distance(@embedding)
          puts "[MDS] stress function after #{@n_iter} iterations: #{@stress}" if @params[:verbose] && (@n_iter % 100).zero?
        end
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, _not_used = nil)
        fit(x)
        @embedding.dup
      end

      # Dump marshal data.
      # @return [Hash] The marshal data.
      def marshal_dump
        { params: @params,
          embedding: @embedding,
          stress: @stress,
          n_iter: @n_iter,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @embedding = obj[:embedding]
        @stress = obj[:stress]
        @n_iter = obj[:n_iter]
        @rng = obj[:rng]
        nil
      end

      private

      def init_embedding(x)
        if @params[:init] == 'pca' && @params[:metric] == 'euclidean'
          pca = Rumale::Decomposition::PCA.new(n_components: @params[:n_components], random_seed: @params[:random_seed])
          pca.fit_transform(x)
        else
          n_samples = x.shape[0]
          sub_rng = @rng.dup
          Rumale::Utils.rand_uniform([n_samples, @params[:n_components]], sub_rng) - 0.5
        end
      end

      def terminate?(old_stress, new_stress)
        return false if @params[:tol].nil?
        return false if old_stress.nil?
        (old_stress - new_stress).abs <= @params[:tol]
      end

      def calc_stress(hi_distance_mat, lo_distance_mat)
        ((hi_distance_mat - lo_distance_mat)**2).sum.fdiv(2)
      end
    end
  end
end
