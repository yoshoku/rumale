# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'
require 'rumale/utils'
require 'rumale/pairwise_metric'
require 'rumale/decomposition/pca'

module Rumale
  # Module for data embedding algorithms.
  module Manifold
    # TSNE is a class that implements t-Distributed Stochastic Neighbor Embedding (t-SNE)
    # with fixed-point optimization algorithm.
    # Fixed-point algorithm usually converges faster than gradient descent method and
    # do not need the learning parameters such as the learning rate and momentum.
    #
    # @example
    #   tsne = Rumale::Manifold::TSNE.new(perplexity: 40.0, init: 'pca', max_iter: 500, random_seed: 1)
    #   representations = tsne.fit_transform(samples)
    #
    # *Reference*
    # - L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," J. of Machine Learning Research, vol. 9, pp. 2579--2605, 2008.
    # - Z. Yang, I. King, Z. Xu, and E. Oja, "Heavy-Tailed Symmetric Stochastic Neighbor Embedding," Proc. NIPS'09, pp. 2169--2177, 2009.
    class TSNE
      include Base::BaseEstimator
      include Base::Transformer

      # Return the data in representation space.
      # @return [Numo::DFloat] (shape: [n_samples, n_components])
      attr_reader :embedding

      # Return the Kullback-Leibler divergence after optimization.
      # @return [Float]
      attr_reader :kl_divergence

      # Return the number of iterations run for optimization
      # @return [Integer]
      attr_reader :n_iter

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer with t-SNE.
      #
      # @param n_components [Integer] The number of dimensions on representation space.
      # @param perplexity [Float] The effective number of neighbors for each point. Perplexity are typically set from 5 to 50.
      # @param metric [String] The metric to calculate the distances in original space.
      #   If metric is 'euclidean', Euclidean distance is calculated for distance in original space.
      #   If metric is 'precomputed', the fit and fit_transform methods expect to be given a distance matrix.
      # @param init [String] The init is a method to initialize the representaion space.
      #   If init is 'random', the representaion space is initialized with normal random variables.
      #   If init is 'pca', the result of principal component analysis as the initial value of the representation space.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of KL-divergence for terminating optimization.
      #   If tol is nil,  it does not use KL divergence as a criterion for terminating the optimization.
      # @param verbose [Boolean] The flag indicating whether to output KL divergence during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_components: 2, perplexity: 30.0, metric: 'euclidean', init: 'random',
                     max_iter: 500, tol: nil, verbose: false, random_seed: nil)
        check_params_integer(n_components: n_components, max_iter: max_iter)
        check_params_float(perplexity: perplexity)
        check_params_string(metric: metric, init: init)
        check_params_boolean(verbose: verbose)
        check_params_type_or_nil(Float, tol: tol)
        check_params_type_or_nil(Integer, random_seed: random_seed)
        check_params_positive(n_components: n_components, perplexity: perplexity, max_iter: max_iter)
        @params = {}
        @params[:n_components] = n_components
        @params[:perplexity] = perplexity
        @params[:max_iter] = max_iter
        @params[:tol] = tol
        @params[:metric] = metric
        @params[:init] = init
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @rng = Random.new(@params[:random_seed])
        @embedding = nil
        @kl_divergence = nil
        @n_iter = nil
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> TSNE
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      #   If the metric is 'precomputed', x must be a square distance matrix (shape: [n_samples, n_samples]).
      # @return [TSNE] The learned transformer itself.
      def fit(x, _not_used = nil)
        check_sample_array(x)
        raise ArgumentError, 'Expect the input distance matrix to be square.' if @params[:metric] == 'precomputed' && x.shape[0] != x.shape[1]
        # initialize some varibales.
        @n_iter = 0
        distance_mat = @params[:metric] == 'precomputed' ? x**2 : Rumale::PairwiseMetric.squared_error(x)
        hi_prob_mat = gaussian_distributed_probability_matrix(distance_mat)
        y = init_embedding(x)
        lo_prob_mat = t_distributed_probability_matrix(y)
        # perform fixed-point optimization.
        one_vec = Numo::DFloat.ones(x.shape[0]).expand_dims(1)
        @params[:max_iter].times do |t|
          break if terminate?(hi_prob_mat, lo_prob_mat)
          a = hi_prob_mat * lo_prob_mat
          b = lo_prob_mat * lo_prob_mat
          y = (b.dot(one_vec) * y + (a - b).dot(y)) / a.dot(one_vec)
          lo_prob_mat = t_distributed_probability_matrix(y)
          @n_iter = t + 1
          if @params[:verbose] && (@n_iter % 100).zero?
            puts "[t-SNE] KL divergence after #{@n_iter} iterations: #{cost(hi_prob_mat, lo_prob_mat)}"
          end
        end
        # store results.
        @embedding = y
        @kl_divergence = cost(hi_prob_mat, lo_prob_mat)
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
          kl_divergence: @kl_divergence,
          n_iter: @n_iter,
          rng: @rng }
      end

      # Load marshal data.
      # @return [nil]
      def marshal_load(obj)
        @params = obj[:params]
        @embedding = obj[:embedding]
        @kl_divergence = obj[:kl_divergence]
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
          Rumale::Utils.rand_normal([n_samples, @params[:n_components]], sub_rng, 0, 0.0001)
        end
      end

      def gaussian_distributed_probability_matrix(distance_mat)
        # initialize some variables.
        n_samples = distance_mat.shape[0]
        prob_mat = Numo::DFloat.zeros(n_samples, n_samples)
        sum_beta = 0.0
        # calculate conditional probabilities.
        n_samples.times do |n|
          beta, probs = optimal_probabilities(n, distance_mat[n, true])
          prob_mat[n, true] = probs
          sum_beta += beta
          puts "[t-SNE] Computed conditional probabilities for sample #{n + 1} / #{n_samples}" if @params[:verbose] && ((n + 1) % 1000).zero?
        end
        puts "[t-SNE] Mean sigma: #{Math.sqrt(n_samples.fdiv(sum_beta))}" if @params[:verbose]
        # symmetrize and normalize probability matrix.
        prob_mat[prob_mat.diag_indices(0)] = 0.0
        prob_mat = 0.5 * (prob_mat + prob_mat.transpose)
        prob_mat / prob_mat.sum
      end

      def optimal_probabilities(sample_id, distance_vec, max_iter = 100)
        # initialize some variables.
        probs = nil
        beta = 1.0
        betamin = Float::MIN
        betamax = Float::MAX
        init_entropy = Math.log(@params[:perplexity])
        # calculate optimal beta and conditional probabilities with binary search.
        max_iter.times do
          entropy, probs = gaussian_distributed_probability_vector(sample_id, distance_vec, beta)
          diff_entropy = entropy - init_entropy
          break if diff_entropy.abs <= 1e-5
          if diff_entropy.positive?
            betamin = beta
            if betamax == Float::MAX
              beta *= 2.0
            else
              beta = 0.5 * (beta + betamax)
            end
          else
            betamax = beta
            if betamin == Float::MIN
              beta /= 2.0
            else
              beta = 0.5 * (beta + betamin)
            end
          end
        end
        [beta, probs]
      end

      def gaussian_distributed_probability_vector(n, distance_vec, beta)
        probs = Numo::NMath.exp(-beta * distance_vec)
        probs[n] = 0.0
        sum_probs = probs.sum
        probs /= sum_probs
        entropy = Math.log(sum_probs) + beta * (distance_vec * probs).sum
        [entropy, probs]
      end

      def t_distributed_probability_matrix(y)
        distance_mat = Rumale::PairwiseMetric.squared_error(y)
        prob_mat = 1.0 / (1.0 + distance_mat)
        prob_mat[prob_mat.diag_indices(0)] = 0.0
        prob_mat / prob_mat.sum
      end

      def cost(p, q)
        (p * Numo::NMath.log(Numo::DFloat.maximum(1e-20, p) / Numo::DFloat.maximum(1e-20, q))).sum
      end

      def terminate?(p, q)
        return false if @params[:tol].nil?
        cost(p, q) <= @params[:tol]
      end
    end
  end
end
