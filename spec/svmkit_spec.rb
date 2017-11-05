require 'spec_helper'

RSpec.describe SVMKit do
  let(:samples) { Marshal.load(File.read(__dir__ + '/test_samples_xor.dat')) }
  let(:labels) { Marshal.load(File.read(__dir__ + '/test_labels_xor.dat')) }
  let(:estimator) do
    SVMKit::LinearModel::SVC.new(reg_param: 1.0, max_iter: 100, batch_size: 20, random_seed: 1)
  end
  let(:transformer) do
    SVMKit::KernelApproximation::RBF.new(gamma: 1.0, n_components: 1024, random_seed: 1)
  end

  it 'classifies xor data.' do
    new_samples = transformer.fit_transform(samples)
    estimator.fit(new_samples, labels)
    score = estimator.score(new_samples, labels)
    expect(score).to eq(1.0)
  end
end
