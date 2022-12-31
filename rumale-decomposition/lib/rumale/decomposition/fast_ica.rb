# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/base/transformer'
require 'rumale/utils'
require 'rumale/validation'

module Rumale
  module Decomposition
    # FastICA is a class that implments Fast Independent Component Analaysis.
    #
    # @example
    #   require 'numo/linalg/autoloader'
    #   require 'rumale/decomposition/fast_ica'
    #
    #   transformer = Rumale::Decomposition::FastICA.new(n_components: 2, random_seed: 1)
    #   source_data = transformer.fit_transform(observed_data)
    #
    # *Reference*
    # - Hyvarinen, A., "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis," IEEE Trans. Neural Networks, Vol. 10 (3), pp. 626--634, 1999.
    # - Hyvarinen, A., and Oja, E., "Independent Component Analysis: Algorithms and Applications," Neural Networks, Vol. 13 (4-5), pp. 411--430, 2000.
    class FastICA < ::Rumale::Base::Estimator
      include ::Rumale::Base::Transformer

      # Returns the unmixing matrix.
      # @return [Numo::DFloat] (shape: [n_components, n_features])
      attr_reader :components

      # Returns the mixing matrix.
      # @return [Numo::DFloat] (shape: [n_features, n_components])
      attr_reader :mixing

      # Returns the number of iterations when converged.
      # @return [Integer]
      attr_reader :n_iter

      # Return the random generator.
      # @return [Random]
      attr_reader :rng

      # Create a new transformer with FastICA.
      #
      # @param n_components [Integer] The number of independent components.
      # @param whiten [Boolean] The flag indicating whether to perform whitening.
      # @param fun [String] The type of contrast function ('logcosh', 'exp', or 'cube').
      # @param alpha [Float] The parameter of contrast function for 'logcosh' and 'exp'.
      #   If fun = 'cube', this parameter is ignored.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param tol [Float] The tolerance of termination criterion.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(n_components: 2, whiten: true, fun: 'logcosh', alpha: 1.0, max_iter: 200, tol: 1e-4, random_seed: nil)
        super()
        @params = {
          n_components: n_components,
          whiten: whiten,
          fun: fun,
          alpha: alpha,
          max_iter: max_iter,
          tol: tol,
          random_seed: (random_seed || srand)
        }
        @rng = Random.new(@params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> FastICA
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @return [FastICA] The learned transformer itself.
      def fit(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        raise 'FastICA#fit requires Numo::Linalg but that is not loaded' unless enable_linalg?(warning: false)

        @mean, whiten_mat = whitening(x, @params[:n_components]) if @params[:whiten]
        wx = @params[:whiten] ? (x - @mean).dot(whiten_mat.transpose) : x
        unmixing, @n_iter = ica(wx, @params[:fun], @params[:max_iter], @params[:tol], @rng.dup)
        @components = @params[:whiten] ? unmixing.dot(whiten_mat) : unmixing
        @mixing = Numo::Linalg.pinv(@components).dup
        if @params[:n_components] == 1
          @components = @components.flatten.dup
          @mixing = @mixing.flatten.dup
        end
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for fitting the model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data
      def fit_transform(x, _y = nil)
        x = ::Rumale::Validation.check_convert_sample_array(x)
        raise 'FastICA#fit_transform requires Numo::Linalg but that is not loaded' unless enable_linalg?(warning: false)

        fit(x).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be transformed with the learned model.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The transformed data.
      def transform(x)
        x = ::Rumale::Validation.check_convert_sample_array(x)

        cx = @params[:whiten] ? (x - @mean) : x
        cx.dot(@components.transpose)
      end

      # Inverse transform the given transformed data with the learned model.
      #
      # @param z [Numo::DFloat] (shape: [n_samples, n_components]) The source data reconstructed to the mixed data.
      # @return [Numo::DFloat] (shape: [n_samples, n_featuress]) The mixed data.
      def inverse_transform(z)
        z = ::Rumale::Validation.check_convert_sample_array(z)

        m = @mixing.shape[1].nil? ? @mixing.expand_dims(0).transpose : @mixing
        x = z.dot(m.transpose)
        x += @mean if @params[:whiten]
        x
      end

      private

      def whitening(x, n_components)
        n_samples, n_features = x.shape
        mean_vec = x.mean(0)
        centered_x = x - mean_vec
        covar_mat = centered_x.transpose.dot(centered_x) / n_samples
        eig_vals, eig_vecs = Numo::Linalg.eigh(covar_mat, vals_range: (n_features - n_components)...n_features)
        [mean_vec, (eig_vecs.reverse(1).dup * (1 / Numo::NMath.sqrt(eig_vals.reverse.dup))).transpose.dup]
      end

      def ica(x, fun, max_iter, tol, sub_rng)
        n_samples, n_components = x.shape
        w = decorrelation(::Rumale::Utils.rand_normal([n_components, n_components], sub_rng))
        n_iters = 0
        max_iter.times do |t|
          n_iters = t + 1
          gx, ggx = gradient(x.dot(w.transpose), fun)
          new_w = decorrelation(gx.transpose.dot(x) / n_samples - w * ggx / n_samples)
          err = (new_w - w).abs.max
          w = new_w
          break if err <= tol
        end
        [w, n_iters]
      end

      def decorrelation(w)
        eig_vals, eig_vecs = Numo::Linalg.eigh(w.dot(w.transpose))
        decorr_mat = (eig_vecs * (1 / Numo::NMath.sqrt(eig_vals))).dot(eig_vecs.transpose)
        decorr_mat.dot(w)
      end

      def gradient(x, func)
        case func
        when 'exp'
          grad_exp(x, @params[:alpha])
        when 'cube'
          grad_cube(x)
        else
          grad_logcosh(x, @params[:alpha])
        end
      end

      def grad_logcosh(x, alpha)
        gx = Numo::NMath.tanh(alpha * x)
        ggx = (alpha * (1 - gx**2)).sum(axis: 0)
        [gx, ggx]
      end

      def grad_exp(x, alpha)
        squared_x = x**2
        exp_x = Numo::NMath.exp(-0.5 * alpha * squared_x)
        gx = exp_x * x
        ggx = (exp_x * (1 - alpha * squared_x)).sum(axis: 0)
        [gx, ggx]
      end

      def grad_cube(x)
        [x**3, (3 * x**2).sum(axis: 0)]
      end
    end
  end
end
