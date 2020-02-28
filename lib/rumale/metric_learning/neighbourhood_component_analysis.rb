# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'
require 'mopti/scaled_conjugate_gradient'

module Rumale
  module MetricLearning
    # NeighbourhoodComponentAnalysis is a class that implements Neighbourhood Component Analysis.
    #
    # @example
    #   transformer = Rumale::MetricLearning::NeighbourhoodComponentAnalysis.new
    #   transformer.fit(training_samples, traininig_labels)
    #   low_samples = transformer.transform(testing_samples)
    #
    # *Reference*
    # - Goldberger, J., Roweis, S., Hinton, G., and Salakhutdinov, R., "Neighbourhood Component Analysis," Advances in NIPS'17, pp. 513--520, 2005.
    class NeighbourhoodComponentAnalysis
      include Base::BaseEstimator
      include Base::Transformer

      # Returns the neighbourhood components.
      # @return [Numo::DFloat] (shape: [n_components, n_features])
      attr_reader :components

      # Return the number of iterations run for optimization
      # @return [Integer]
      attr_reader :n_iter

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer with NeighbourhoodComponentAnalysis.
      #
      # @param n_components [Integer] The number of components.
      # @param init [String] The initialization method for components ('random' or 'pca').
      # @param tol [Float] The tolerance of termination criterion.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_components: nil, init: 'random', tol: 1e-6, verbose: false, random_seed: nil)
        check_params_numeric_or_nil(n_components: n_components, random_seed: random_seed)
        check_params_numeric(tol: tol)
        check_params_string(init: init)
        check_params_boolean(verbose: verbose)
        @params = {}
        @params[:n_components] = n_components
        @params[:init] = init
        @params[:tol] = tol
        @params[:verbose] = verbose
        @params[:random_seed] = random_seed
        @params[:random_seed] ||= srand
        @components = nil
        @n_iter = nil
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [NeighbourhoodComponentAnalysis] The learned classifier itself.
      def fit(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        check_sample_label_size(x, y)
        n_features = x.shape[1]
        n_components = if @params[:n_components].nil?
                         n_features
                       else
                         [n_features, @params[:n_components]].min
                       end
        @components, @n_iter = optimize_components(x, y, n_features, n_components)
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @param y [Numo::Int32] (shape: [n_samples]) The labels to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, y)
        x = check_convert_sample_array(x)
        y = check_convert_label_array(y)
        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = check_convert_sample_array(x)
        x.dot(@components.transpose)
      end

      private

      def init_components(x, n_features, n_components)
        if @params[:init] == 'pca'
          pca = Rumale::Decomposition::PCA.new(n_components: n_components, solver: 'evd')
          pca.fit(x).components.flatten.dup
        else
          Rumale::Utils.rand_normal([n_features, n_components], @rng.dup).flatten.dup
        end
      end

      def optimize_components(x, y, n_features, n_components)
        # initialize components.
        comp_init = init_components(x, n_features, n_components)
        # initialize optimization results.
        res = {}
        res[:x] = comp_init
        res[:n_iter] = 0
        # perform optimization.
        optimizer = Mopti::ScaledConjugateGradient.new(
          fnc: method(:nca_loss), jcb: method(:nca_dloss), x_init: comp_init, args: [x, y], ftol: @params[:tol]
        )
        fold = 0.0
        dold = 0.0
        optimizer.each do |prm|
          res = prm
          puts "[NeighbourhoodComponentAnalysis] Loss after #{res[:n_iter]} epochs: #{n_samples - res[:fnc]}" if @params[:verbose]
          break if (fold - res[:fnc]).abs <= @params[:tol] && (dold - res[:jcb]).abs <= @params[:tol]
          fold = res[:fnc]
          dold = res[:jcb]
        end
        # return the results.
        n_iter = res[:n_iter]
        comps = n_components == 1 ? res[:x].dup : res[:x].reshape(n_components, n_features)
        [comps, n_iter]
      end

      def nca_loss(w, x, y)
        # initialize some variables.
        n_samples, n_features = x.shape
        n_components = w.size / n_features
        # projection.
        w = w.reshape(n_components, n_features)
        z = x.dot(w.transpose)
        # calculate probability matrix.
        prob_mat = probability_matrix(z)
        # calculate loss.
        # NOTE:
        # NCA attempts to maximize its objective function.
        # For the minization algorithm, the objective function value is subtracted from the maixmum value (n_samples).
        mask_mat = y.expand_dims(1).eq(y)
        masked_prob_mat = prob_mat * mask_mat
        n_samples - masked_prob_mat.sum
      end

      def nca_dloss(w, x, y)
        # initialize some variables.
        n_features = x.shape[1]
        n_components = w.size / n_features
        # projection.
        w = w.reshape(n_components, n_features)
        z = x.dot(w.transpose)
        # calculate probability matrix.
        prob_mat = probability_matrix(z)
        # calculate gradient.
        mask_mat = y.expand_dims(1).eq(y)
        masked_prob_mat = prob_mat * mask_mat
        weighted_prob_mat = masked_prob_mat - prob_mat * masked_prob_mat.sum(1).expand_dims(1)
        weighted_prob_mat += weighted_prob_mat.transpose
        weighted_prob_mat[weighted_prob_mat.diag_indices] = -weighted_prob_mat.sum(0)
        gradient = 2 * z.transpose.dot(weighted_prob_mat).dot(x)
        -gradient.flatten.dup
      end

      def probability_matrix(z)
        prob_mat = Numo::NMath.exp(-Rumale::PairwiseMetric.squared_error(z))
        prob_mat[prob_mat.diag_indices] = 0.0
        prob_mat /= prob_mat.sum(1).expand_dims(1)
        prob_mat
      end
    end
  end
end
