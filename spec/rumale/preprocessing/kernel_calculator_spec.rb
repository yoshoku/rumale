# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::KernelCalculator do
  describe 'nonlinear classification problem' do
    let(:dataset) { xor_dataset }
    let(:x) { dataset[0] }
    let(:y) { dataset[1] }
    let(:res) { Rumale::ModelSelection.train_test_split(x, y, test_size: 0.1, random_seed: 1) }
    let(:x_train) { res[0] }
    let(:x_test) { res[1] }
    let(:y_train) { res[2] }
    let(:y_test) { res[3] }
    let(:classifier) do
      Rumale::Pipeline::Pipeline.new(
        steps: {
          kcal: described_class.new,
          ksvc: Rumale::KernelMachine::KernelSVC.new(random_seed: 0)
        }
      ).fit(x_train, y_train)
    end

    let(:calculator) { classifier.steps[:kcal] }
    let(:copied) { Marshal.load(Marshal.dump(calculator)) }

    it 'classifies xor dataset', :aggregate_failures do
      expect(calculator).to be_a(described_class)
      expect(calculator.params).to match({ kernel: 'rbf', kernel_params: nil })
      expect(calculator.components).to be_a(Numo::DFloat)
      expect(calculator.components.ndim).to eq(2)
      expect(calculator.components.shape).to eq(x_train.shape)
      expect(classifier.score(x_test, y_test)).to be_within(0.05).of(1.0)
    end

    it 'dumps and restores itself using Marshal module', :aggregate_failures do
      expect(copied).to be_a(described_class)
      expect(copied.params).to eq(calculator.params)
      expect(copied.components).to eq(calculator.components)
    end
  end

  describe 'kernel type parameter' do
    shared_examples 'kernel matrix' do
      it 'calculates kernel matrix', :aggregate_failures do
        expect(kernel_mat_train).to be_a(Numo::DFloat)
        expect(kernel_mat_train.ndim).to eq(2)
        expect(kernel_mat_train.shape[0]).to eq(3)
        expect(kernel_mat_train.shape[1]).to eq(3)
        expect(kernel_mat_test).to be_a(Numo::DFloat)
        expect(kernel_mat_test.ndim).to eq(2)
        expect(kernel_mat_test.shape[0]).to eq(2)
        expect(kernel_mat_test.shape[1]).to eq(3)
      end
    end

    let(:kernel_params) { nil }
    let(:calculator) { described_class.new(kernel: kernel, kernel_params: kernel_params) }
    let(:x_train) { Numo::DFloat[[1, 2], [3, 2], [4, 3]] }
    let(:x_test) { Numo::DFloat[[2, 4], [3, 2]] }
    let(:kernel_mat_train) { calculator.fit_transform(x_train) }
    let(:kernel_mat_test) { calculator.fit(x_train).transform(x_test) }

    context "when kernel is 'rbf'" do
      let(:kernel) { 'rbf' }

      it_behaves_like 'kernel matrix'
    end

    context "when kernel is 'linear'" do
      let(:kernel) { 'linear' }

      it_behaves_like 'kernel matrix'
    end

    context "when kernel is 'poly'" do
      let(:kernel) { 'poly' }

      it_behaves_like 'kernel matrix'
    end

    context "when kernel is 'sigmoid'" do
      let(:kernel) { 'sigmoid' }
      let(:kernel_params) { { gamma: 0.01, coef: 0 } }

      it_behaves_like 'kernel matrix'
    end

    context 'when kernel is user defined' do
      let(:kernel) do
        proc do |x, y, b:|
          x.dot(y.transpose) + b
        end
      end

      let(:kernel_params) { { b: 100 } }

      it_behaves_like 'kernel matrix'
    end

    context 'when wrong kernel type is given' do
      let(:kernel) { 'foo' }

      it 'raises ArgumentError' do
        expect { calculator.fit_transform(x_train) }.to raise_error(ArgumentError)
        expect { calculator.fit(x_train).transform(x_test) }.to raise_error(ArgumentError)
      end
    end
  end
end
