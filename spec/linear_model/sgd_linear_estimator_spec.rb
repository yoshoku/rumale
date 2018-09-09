# frozen_string_literal: true

require 'spec_helper'

RSpec.describe SVMKit::LinearModel::SGDLinearEstimator do
  let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples.dat')) }
  let(:y) { x.dot(Numo::DFloat[1.0, 2.0]) }
  let(:estimator) { described_class.new(random_seed: 1) }

  it 'raises NotImplementedError when calls partial_fit method.' do
    expect { estimator.send(:partial_fit, x, y) }.to raise_error(NotImplementedError)
  end

  it 'initializes some parameters.' do
    expect(estimator.params[:reg_param]).to eq(1.0)
    expect(estimator.params[:fit_bias]).to be_falsy
    expect(estimator.params[:bias_scale]).to eq(1.0)
    expect(estimator.params[:max_iter]).to eq(1000)
    expect(estimator.params[:batch_size]).to eq(10)
    expect(estimator.params[:optimizer].class).to eq(SVMKit::Optimizer::Nadam)
    expect(estimator.params[:random_seed]).to eq(1)
  end
end
