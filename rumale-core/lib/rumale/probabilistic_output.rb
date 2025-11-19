# frozen_string_literal: true

require 'numo/narray/alt'

module Rumale
  # Module for calculating posterior class probabilities with SVM outputs.
  # This module is used for internal processes.
  #
  # @example
  #   estimator = Rumale::LinearModel::SVC.new
  #   estimator.fit(x, bin_y)
  #   df = estimator.decision_function(x)
  #   params = Rumale::ProbabilisticOutput.fit_sigmoid(df, bin_y)
  #   probs = 1 / (Numo::NMath.exp(params[0] * df + params[1]) + 1)
  #
  # *Reference*
  # - Platt, J C., "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods," Adv. Large Margin Classifiers, pp. 61--74, 2000.
  # - Lin, H-T., Lin, C-J., and Weng, R C., "A Note on Platt's Probabilistic Outputs for Support Vector Machines," J. Machine Learning, Vol. 63 (3), pp. 267--276, 2007.
  module ProbabilisticOutput
    class << self
      # Fit the probabilistic model for binary SVM outputs.
      #
      # @param df [Numo::DFloat] (shape: [n_samples]) The outputs of decision function to be used for fitting the model.
      # @param bin_y [Numo::Int32] (shape: [n_samples]) The binary labels to be used for fitting the model.
      # @param max_iter [Integer] The maximum number of iterations.
      # @param min_step [Float] The minimum step of Newton's method.
      # @param sigma [Float] The parameter to avoid hessian matrix from becoming singular matrix.
      # @return [Numo::DFloat] (shape: 2) The parameters of the model.
      def fit_sigmoid(df, bin_y, max_iter = 100, min_step = 1e-10, sigma = 1e-12)
        # Initialize some variables.
        n_samples = bin_y.size
        negative_label = bin_y.to_a.uniq.min
        pos = bin_y.ne(negative_label)
        neg = bin_y.eq(negative_label)
        n_pos_samples = pos.count
        n_neg_samples = neg.count
        target_probs = Numo::DFloat.zeros(n_samples)
        target_probs[pos] = (n_pos_samples + 1) / (n_pos_samples + 2.0)
        target_probs[neg] = 1 / (n_neg_samples + 2.0)
        alpha = 0.0
        beta = Math.log((n_neg_samples + 1) / (n_pos_samples + 1.0))
        err = error_function(target_probs, df, alpha, beta)
        # Optimize parameters for class porbability calculation.
        old_grad_vec = Numo::DFloat.zeros(2)
        max_iter.times do
          # Calculate gradient and hessian matrix.
          probs = predicted_probs(df, alpha, beta)
          grad_vec = gradient(target_probs, probs, df)
          hess_mat = hessian_matrix(probs, df, sigma)
          break if grad_vec.abs.lt(1e-5).count == 2
          break if (old_grad_vec - grad_vec).abs.sum < 1e-5

          old_grad_vec = grad_vec
          # Calculate Newton directions.
          dirs_vec = directions(grad_vec, hess_mat)
          grad_dir = grad_vec.dot(dirs_vec)
          stepsize = 2.0
          while stepsize >= min_step
            stepsize *= 0.5
            new_alpha = alpha + stepsize * dirs_vec[0]
            new_beta = beta + stepsize * dirs_vec[1]
            new_err = error_function(target_probs, df, new_alpha, new_beta)
            next unless new_err < err + 0.0001 * stepsize * grad_dir

            alpha = new_alpha
            beta = new_beta
            err = new_err
            break
          end
        end
        Numo::DFloat[alpha, beta]
      end

      private

      def error_function(target_probs, df, alpha, beta)
        fn = alpha * df + beta
        pos = fn.ge(0.0)
        neg = fn.lt(0.0)
        err = 0.0
        err += (target_probs[pos] * fn[pos] + Numo::NMath.log(1 + Numo::NMath.exp(-fn[pos]))).sum if pos.count.positive?
        err += ((target_probs[neg] - 1) * fn[neg] + Numo::NMath.log(1 + Numo::NMath.exp(fn[neg]))).sum if neg.count.positive?
        err
      end

      def predicted_probs(df, alpha, beta)
        fn = alpha * df + beta
        pos = fn.ge(0.0)
        neg = fn.lt(0.0)
        probs = Numo::DFloat.zeros(df.shape[0])
        probs[pos] = Numo::NMath.exp(-fn[pos]) / (1 + Numo::NMath.exp(-fn[pos])) if pos.count.positive?
        probs[neg] = 1 / (1 + Numo::NMath.exp(fn[neg])) if neg.count.positive?
        probs
      end

      def gradient(target_probs, probs, df)
        sub = target_probs - probs
        Numo::DFloat[(df * sub).sum, sub.sum]
      end

      def hessian_matrix(probs, df, sigma)
        sub = probs * (1 - probs)
        h11 = (df**2 * sub).sum + sigma
        h22 = sub.sum + sigma
        h21 = (df * sub).sum
        Numo::DFloat[[h11, h21], [h21, h22]]
      end

      def directions(grad_vec, hess_mat)
        det = hess_mat[0, 0] * hess_mat[1, 1] - hess_mat[0, 1] * hess_mat[1, 0]
        inv_hess_mat = Numo::DFloat[[hess_mat[1, 1], -hess_mat[0, 1]], [-hess_mat[1, 0], hess_mat[0, 0]]] / det
        -inv_hess_mat.dot(grad_vec)
      end
    end
  end
end
