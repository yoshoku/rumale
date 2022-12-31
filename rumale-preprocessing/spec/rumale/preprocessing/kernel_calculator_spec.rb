# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Preprocessing::KernelCalculator do
  describe 'kernel type parameter' do
    let(:gamma) { 1 }
    let(:degree) { 3 }
    let(:coef) { 1 }
    let(:calculator) { described_class.new(kernel: kernel, gamma: gamma, degree: degree, coef: coef) }
    let(:x_train) { Numo::DFloat[[1, 2], [3, 2], [4, 3]] }
    let(:x_test) { Numo::DFloat[[2, 4], [3, 2]] }

    shared_examples 'kernel matrix' do
      let(:kernel_mat_train) { calculator.fit_transform(x_train) }
      let(:kernel_mat_test) { calculator.fit(x_train).transform(x_test) }

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
      let(:gamma) { 0.01 }
      let(:coef) { 0 }

      it_behaves_like 'kernel matrix'
    end

    context 'when wrong kernel type is given' do
      let(:kernel) { 'foo' }

      it 'raises ArgumentError', :aggregate_failures do
        expect { calculator.fit_transform(x_train) }.to raise_error(ArgumentError)
        expect { calculator.fit(x_train).transform(x_test) }.to raise_error(ArgumentError)
      end
    end
  end
end
