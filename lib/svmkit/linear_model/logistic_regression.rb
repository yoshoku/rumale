require 'svmkit/base/base_estimator'
require 'svmkit/base/classifier'

module SVMKit
  # This module consists of the classes that implement generalized linear models.
  module LinearModel
    # LogisticRegression is a class that implements Logistic Regression with stochastic gradient descent (SGD) optimization.
    # Note that the Logistic Regression of SVMKit performs as a binary classifier.
    #
    #   estimator =
    #     SVMKit::LinearModel::LogisticRegression.new(reg_param: 1.0, max_iter: 100, batch_size: 20, random_seed: 1)
    #   estimator.fit(training_samples, traininig_labels)
    #   results = estimator.predict(testing_samples)
    #
    # * *Reference*:
    #   - S. Shalev-Shwartz, Y. Singer, N. Srebro, and A. Cotter, "Pegasos: Primal Estimated sub-GrAdient SOlver for SVM," Mathematical Programming, vol. 127 (1), pp. 3--30, 2011.
    #
    class LogisticRegression
      include Base::BaseEstimator
      include Base::Classifier

      DEFAULT_PARAMS = { # :nodoc:
        reg_param: 1.0,
        max_iter: 100,
        batch_size: 50,
        random_seed: nil
      }.freeze

      # The weight vector for Logistic Regression.
      attr_reader :weight_vec

      # The random generator for performing random sampling in the SGD optimization.
      attr_reader :rng

      # Create a new classifier with Logisitc Regression by the SGD optimization.
      #
      # :call-seq:
      #   new(reg_param: 1.0, max_iter: 100, batch_size: 50, random_seed: 1) -> LogisiticRegression
      #
      # * *Arguments* :
      #   - +:reg_param+ (Float) (defaults to: 1.0) -- The regularization parameter.
      #   - +:max_iter+ (Integer) (defaults to: 100) -- The maximum number of iterations.
      #   - +:batch_size+ (Integer) (defaults to: 50) -- The size of the mini batches.
      #   - +:random_seed+ (Integer) (defaults to: nil) -- The seed value using to initialize the random generator.
      def initialize(params = {})
        self.params = DEFAULT_PARAMS.merge(Hash[params.map { |k, v| [k.to_sym, v] }])
        self.params[:random_seed] ||= srand
        @weight_vec = nil
        @rng = Random.new(self.params[:random_seed])
      end

      # Fit the model with given training data.
      #
      # :call-seq:
      #   fit(x, y) -> LogisticRegression
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The training data to be used for fitting the model.
      #   - +y+ (NMatrix, shape: [1, n_samples]) -- The categorical variables (e.g. labels) to be used for fitting the model.
      # * *Returns* :
      #   - The learned classifier itself.
      def fit(x, y)
        # Generate binary labels
        negative_label = y.uniq.sort.shift
        bin_y = y.to_flat_a.map { |l| l != negative_label ? 1 : 0 }
        # Initialize some variables.
        n_samples, n_features = x.shape
        rand_ids = [*0..n_samples - 1].shuffle(random: @rng)
        @weight_vec = NMatrix.zeros([1, n_features])
        # Start optimization.
        params[:max_iter].times do |t|
          # random sampling
          subset_ids = rand_ids.shift(params[:batch_size])
          rand_ids.concat(subset_ids)
          # update the weight vector.
          eta = 1.0 / (params[:reg_param] * (t + 1))
          mean_vec = NMatrix.zeros([1, n_features])
          subset_ids.each do |n|
            z = @weight_vec.dot(x.row(n).transpose)[0]
            coef = bin_y[n] / (1.0 + Math.exp(bin_y[n] * z))
            mean_vec += x.row(n) * coef
          end
          mean_vec *= eta / params[:batch_size]
          @weight_vec = @weight_vec * (1.0 - eta * params[:reg_param]) + mean_vec
          # scale the weight vector.
          scaler = (1.0 / params[:reg_param]**0.5) / @weight_vec.norm2
          @weight_vec *= [1.0, scaler].min
        end
        self
      end

      # Calculate confidence scores for samples.
      #
      # :call-seq:
      #   decision_function(x) -> NMatrix, shape: [1, n_samples]
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to compute the scores.
      # * *Returns* :
      #   - Confidence score per sample.
      def decision_function(x)
        w = (@weight_vec.dot(x.transpose) * -1.0).exp + 1.0
        w.map { |v| 1.0 / v }
      end

      # Predict class labels for samples.
      #
      # :call-seq:
      #   predict(x) -> NMatrix, shape: [1, n_samples]
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to predict the labels.
      # * *Returns* :
      #   - Predicted class label per sample.
      def predict(x)
        decision_function(x).map { |v| v >= 0.5 ? 1 : -1 }
      end

      # Predict probability for samples.
      #
      # :call-seq:
      #   predict_proba(x) -> NMatrix, shape: [1, n_samples]
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- The samples to predict the probailities.
      # * *Returns* :
      #   - Predicted probability per sample.
      def predict_proba(x)
        decision_function(x)
      end

      # Claculate the mean accuracy of the given testing data.
      #
      # :call-seq:
      #   score(x, y) -> Float
      #
      # * *Arguments* :
      #   - +x+ (NMatrix, shape: [n_samples, n_features]) -- Testing data.
      #   - +y+ (NMatrix, shape: [1, n_samples]) -- True labels for testing data.
      # * *Returns* :
      #   - Mean accuracy
      def score(x, y)
        p = predict(x)
        n_hits = (y.to_flat_a.map.with_index { |l, n| l == p[n] ? 1 : 0 }).inject(:+)
        n_hits / y.size.to_f
      end

      # Serializes object through Marshal#dump.
      def marshal_dump # :nodoc:
        { params: params, weight_vec: Utils.dump_nmatrix(@weight_vec), rng: @rng }
      end

      # Deserialize object through Marshal#load.
      def marshal_load(obj) # :nodoc:
        self.params = obj[:params]
        @weight_vec = Utils.restore_nmatrix(obj[:weight_vec])
        @rng = obj[:rng]
        nil
      end
    end
  end
end
