# frozen_string_literal: true

require 'rumale/base/base_estimator'
require 'rumale/base/transformer'
require 'rumale/pairwise_metric'

module Rumale
  module Preprocessing
    # KernelCalculator is a class that calculates the kernel matrix with training data.
    #
    # @example
    #   transformer = Rumale::Preprocessing::KernelCalculator.new(
    #     kernel: 'rbf', kernel_params: { gamma: 0.5 }
    #   )
    #   regressor = Rumale::KernelMachine::KernelRidge.new
    #   pipeline = Rumale::Pipeline::Pipeline.new(
    #     steps: { trs: transfomer, est: regressor }
    #   )
    #   pipeline.fit(x_train, y_train)
    #   results = pipeline.predict(x_test)
    class KernelCalculator
      include Base::BaseEstimator
      include Base::Transformer

      # Returns the training data for calculating kernel matrix.
      # @return [Numo::DFloat] (shape: n_components, n_features)
      attr_reader :components

      # Create a new transformer that transforms feature vectors into a kernel matrix.
      #
      # @param kernel [String/Method/Proc] The type of kernel function ('rbf', 'linear', 'poly', 'sigmoid', and user defined method).
      # @param kernel_params [Hash/Nil] The parameters of kernel function. If nil is given, the fallowing default values will be used.
      #   'rbf': { gamma: 1.0 }
      #   'linear': nil
      #   'poly': { degree: 3, gamma: 1, coef: 1 }
      #   'sigmoid': { gamma: 1, coef: 1 }
      def initialize(kernel: 'rbf', kernel_params: nil)
        check_params_type_or_nil(Hash, kernel_params: kernel_params)
        @params = {}
        @params[:kernel] = kernel
        @params[:kernel_params] = kernel_params
        @components = nil
      end

      # Fit the model with given training data.
      #
      # @overload fit(x) -> KernelCalculator
      #   @param x [Numo::NArray] (shape: [n_samples, n_features]) The training data to be used for calculating kernel matrix.
      # @return [KernelCalculator] The learned transformer itself.
      def fit(x, _y = nil)
        x = check_convert_sample_array(x)
        @components = x.dup
        self
      end

      # Fit the model with training data, and then transform them with the learned model.
      #
      # @overload fit_transform(x) -> Numo::DFloat
      #   @param x [Numo::DFloat] (shape: [n_samples, n_features]) The training data to be used for calculating kernel matrix.
      # @return [Numo::DFloat] (shape: [n_samples, n_samples]) The calculated kernel matrix.
      def fit_transform(x, y = nil)
        x = check_convert_sample_array(x)
        fit(x, y).transform(x)
      end

      # Transform the given data with the learned model.
      #
      # @param x [Numo::DFloat] (shape: [n_samples, n_features]) The data to be used for calculating kernel matrix with the training data.
      # @return [Numo::DFloat] (shape: [n_samples, n_components]) The calculated kernel matrix.
      def transform(x)
        x = check_convert_sample_array(x)
        kernel_mat(kernel_fnc, x, @params[:kernel_params])
      end

      private

      def kernel_fnc
        return @params[:kernel] if @params[:kernel].is_a?(Method) || @params[:kernel].is_a?(Proc)

        case @params[:kernel]
        when 'rbf'
          proc do |x, y, gamma: 1|
            Numo::NMath.exp(-gamma * Rumale::PairwiseMetric.squared_error(x, y))
          end
        when 'poly'
          proc do |x, y, degree: 3, gamma: 1, coef: 1|
            (x.dot(y.transpose) * gamma + coef)**degree
          end
        when 'sigmoid'
          proc do |x, y, gamma: 1, coef: 1|
            Numo::NMath.tanh(gamma * x.dot(y.transpose) + coef)
          end
        when 'linear'
          proc do |x, y|
            x.dot(y.transpose)
          end
        else
          raise ArgumentError, ''
        end
      end

      def kernel_mat(fnc, x, args)
        if args.is_a?(Hash)
          fnc.call(x, @components, **args)
        else
          fnc.call(x, @components)
        end
      end
    end
  end
end
