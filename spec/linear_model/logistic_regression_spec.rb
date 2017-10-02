require 'spec_helper'

RSpec.describe SVMKit::LinearModel::LogisticRegression do
  let(:samples) { SVMKit::Utils.restore_nmatrix(Marshal.load(File.read(__dir__ + '/test_samples.dat'))) }
  let(:labels) { SVMKit::Utils.restore_nmatrix(Marshal.load(File.read(__dir__ + '/test_labels.dat'))) }
  let(:estimator) { described_class.new(penalty: 1.0, max_iter: 100, batch_size: 20, random_seed: 1) }

  it 'classifies two clusters.' do
    estimator.fit(samples, labels)
    score = estimator.score(samples, labels)
    expect(score).to eq(1.0)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(samples, labels)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.params[:reg_param]).to eq(copied.params[:reg_param])
    expect(estimator.params[:max_iter]).to eq(copied.params[:max_iter])
    expect(estimator.params[:batch_size]).to eq(copied.params[:batch_size])
    expect(estimator.params[:random_seed]).to eq(copied.params[:random_seed])
    expect(estimator.weight_vec).to eq(copied.weight_vec)
    expect(estimator.rng).to eq(copied.rng)
  end
end
