# frozen_string_literal: true

require 'rumale/base/estimator'
require 'rumale/utils'

module Rumale
  module NeuralNetwork
    # @!visibility private
    # This module consists of the classes that implement layer functions of neural network.
    module Layer
      # @!visibility private
      # Affine is a class that calculates the linear transform.
      # This class is used internally.
      class Affine
        # @!visibility private
        def initialize(n_inputs: nil, n_outputs: nil, optimizer: nil, rng: nil)
          @weight = 0.01 * ::Rumale::Utils.rand_normal([n_inputs, n_outputs], rng)
          @bias = Numo::DFloat.zeros(n_outputs)
          @optimizer_weight = optimizer.dup
          @optimizer_bias = optimizer.dup
        end

        # @!visibility private
        def forward(x)
          out = x.dot(@weight) + @bias

          backward = proc do |dout|
            dx = dout.dot(@weight.transpose)
            dw = x.transpose.dot(dout)
            db = dout.sum(axis: 0)

            @weight = @optimizer_weight.call(@weight, dw)
            @bias = @optimizer_bias.call(@bias, db)

            dx
          end

          [out, backward]
        end
      end

      # @!visibility private
      # Dropout is a class that performs dropout regularization.
      # This class is used internally.
      class Dropout
        # @!visibility private
        def initialize(rate: 0.3, rng: nil)
          @rate = rate
          @rng = rng
        end

        # @!visibility private
        def forward(x)
          rand_mat = ::Rumale::Utils.rand_uniform(x.shape, @rng)
          mask = rand_mat.ge(@rate)
          out = x * mask
          out *= 1.fdiv(1 - @rate) if @rate < 1.0

          backward = proc { |dout| dout * mask }

          [out, backward]
        end
      end

      # @!visibility private
      # ReLU is a class that calculates rectified linear function.
      # This class is used internally.
      class Relu
        # @!visibility private
        def forward(x)
          mask = x.gt(0)
          out = x * mask

          backward = proc { |dout| dout * mask }

          [out, backward]
        end
      end
    end

    # @!visibility private
    # This module consists of the classes that implement loss function for neural network.
    module Loss
      # @!visibility private
      # MeanSquaredError is a class that calculates mean squared error for regression task.
      # This class is used internally.
      class MeanSquaredError
        # @!visibility private
        def call(out, y)
          sz_batch = y.shape[0]
          diff = out - y
          loss = (diff**2).sum.fdiv(sz_batch)
          dout = 2.fdiv(sz_batch) * diff
          [loss, dout]
        end
      end

      # @!visibility private
      # SoftmaxCrossEntropy is a class that calculates softmax cross-entropy for classification task.
      # This class is used internally.
      class SoftmaxCrossEntropy
        # @!visibility private
        def call(out, y)
          sz_batch = y.shape[0]
          z = softmax(out)
          loss = -(y * Numo::NMath.log(z + 1e-8)).sum.fdiv(sz_batch)
          dout = (z - y) / sz_batch
          [loss, dout]
        end

        private

        def softmax(x)
          clip = x.max(-1).expand_dims(-1)
          exp_x = Numo::NMath.exp(x - clip)
          exp_x / exp_x.sum(axis: -1).expand_dims(-1)
        end
      end
    end

    # @!visibility private
    # This module consists of the classes for implementing neural network model.
    module Model
      # @!visibility private
      attr_reader :layers

      # @!visibility private
      # Sequential is a class that implements linear stack model.
      # This class is used internally.
      class Sequential
        # @!visibility private
        def initialize
          @layers = []
        end

        # @!visibility private
        def push(ops)
          @layers.push(ops)
          self
        end

        # @!visibility private
        def delete_dropout
          @layers.delete_if { |node| node.is_a?(Layer::Dropout) }
          self
        end

        # @!visibility private
        def forward(x)
          backprops = []
          out = x.dup

          @layers.each do |l|
            out, bw = l.forward(out)
            backprops.push(bw)
          end

          backward = proc do |dout|
            backprops.reverse_each { |bw| dout = bw.call(dout) }
            dout
          end

          [out, backward]
        end
      end
    end

    # @!visibility private
    # This module consists of the classes that implement optimizers adaptively tuning learning rate.
    module Optimizer
      # @!visibility private
      # Adam is a class that implements Adam optimizer.
      #
      # *Reference*
      # - Kingma, D P., and Ba, J., "Adam: A Method for Stochastic Optimization," Proc. ICLR'15, 2015.
      class Adam
        # @!visibility private
        # Create a new optimizer with Adam
        #
        # @param learning_rate [Float] The initial value of learning rate.
        # @param decay1 [Float] The smoothing parameter for the first moment.
        # @param decay2 [Float] The smoothing parameter for the second moment.
        def initialize(learning_rate: 0.001, decay1: 0.9, decay2: 0.999)
          @params = {
            learning_rate: learning_rate,
            decay1: decay1,
            decay2: decay2
          }
          @iter = 0
        end

        # @!visibility private
        # Calculate the updated weight with Adam adaptive learning rate.
        #
        # @param weight [Numo::DFloat] (shape: [n_features]) The weight to be updated.
        # @param gradient [Numo::DFloat] (shape: [n_features]) The gradient for updating the weight.
        # @return [Numo::DFloat] (shape: [n_feautres]) The updated weight.
        def call(weight, gradient)
          @fst_moment ||= Numo::DFloat.zeros(weight.shape)
          @sec_moment ||= Numo::DFloat.zeros(weight.shape)

          @iter += 1

          @fst_moment = @params[:decay1] * @fst_moment + (1.0 - @params[:decay1]) * gradient
          @sec_moment = @params[:decay2] * @sec_moment + (1.0 - @params[:decay2]) * gradient**2
          nm_fst_moment = @fst_moment / (1.0 - @params[:decay1]**@iter)
          nm_sec_moment = @sec_moment / (1.0 - @params[:decay2]**@iter)

          weight - @params[:learning_rate] * nm_fst_moment / (nm_sec_moment**0.5 + 1e-8)
        end
      end
    end

    # BaseMLP is an abstract class for implementation of multi-layer peceptron estimator.
    # This class is used internally.
    class BaseMLP < ::Rumale::Base::Estimator
      # Create a multi-layer perceptron estimator.
      #
      # @param hidden_units [Array] The number of units in the i-th hidden layer.
      # @param dropout_rate [Float] The rate of the units to drop.
      # @param learning_rate [Float] The initial value of learning rate in Adam optimizer.
      # @param decay1 [Float] The smoothing parameter for the first moment in Adam optimizer.
      # @param decay2 [Float] The smoothing parameter for the second moment in Adam optimizer.
      # @param max_iter [Integer] The maximum number of epochs that indicates
      #   how many times the whole data is given to the training process.
      # @param batch_size [Intger] The size of the mini batches.
      # @param tol [Float] The tolerance of loss for terminating optimization.
      # @param verbose [Boolean] The flag indicating whether to output loss during iteration.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def initialize(hidden_units: [128, 128], dropout_rate: 0.4, learning_rate: 0.001, decay1: 0.9, decay2: 0.999,
                     max_iter: 200, batch_size: 50, tol: 1e-4, verbose: false, random_seed: nil)
        super()
        @params = {
          hidden_units: hidden_units,
          dropout_rate: dropout_rate,
          learning_rate: learning_rate,
          decay1: decay1,
          decay2: decay2,
          max_iter: max_iter,
          batch_size: batch_size,
          tol: tol,
          verbose: verbose,
          random_seed: random_seed || srand
        }
        @rng = Random.new(@params[:random_seed])
      end

      private

      def buld_network(n_inputs, n_outputs, srng = nil)
        adam = ::Rumale::NeuralNetwork::Optimizer::Adam.new(
          learning_rate: @params[:learning_rate], decay1: @params[:decay1], decay2: @params[:decay2]
        )
        model = ::Rumale::NeuralNetwork::Model::Sequential.new
        n_units = [n_inputs, *@params[:hidden_units]]
        n_units.each_cons(2) do |n_in, n_out|
          model.push(::Rumale::NeuralNetwork::Layer::Affine.new(n_inputs: n_in, n_outputs: n_out, optimizer: adam, rng: srng))
          model.push(::Rumale::NeuralNetwork::Layer::Relu.new)
          model.push(::Rumale::NeuralNetwork::Layer::Dropout.new(rate: @params[:dropout_rate], rng: srng))
        end
        model.push(::Rumale::NeuralNetwork::Layer::Affine.new(n_inputs: n_units[-1], n_outputs: n_outputs, optimizer: adam,
                                                              rng: srng))
      end

      def train(x, y, network, loss_func, srng = nil)
        class_name = self.class.to_s.split('::').last
        n_samples = x.shape[0]

        @params[:max_iter].times do |t|
          sample_ids = Array(0...n_samples)
          sample_ids.shuffle!(random: srng)
          until (subset_ids = sample_ids.shift(@params[:batch_size])).empty?
            # random sampling
            sub_x = x[subset_ids, true].dup
            sub_y = y[subset_ids, true].dup
            # forward
            out, backward = network.forward(sub_x)
            # calc loss function
            loss, dout = loss_func.call(out, sub_y)
            break if loss < @params[:tol]

            # backward
            backward.call(dout)
          end
          @n_iter = t + 1
          puts "[#{class_name}] Loss after #{@n_iter} epochs: #{loss}" if @params[:verbose]
        end

        network
      end
    end
  end
end
