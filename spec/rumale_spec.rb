# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale do
  let(:xor) { xor_dataset }
  let(:samples) { xor[0] }
  let(:labels) { xor[1] }
  let(:estimator) { Rumale::LinearModel::SVC.new(reg_param: 0.01, random_seed: 1) }
  let(:transformer) { Rumale::KernelApproximation::RBF.new(gamma: 1.0, n_components: 1024, random_seed: 1) }

  it 'classifies xor data.' do
    new_samples = transformer.fit_transform(samples)
    estimator.fit(new_samples, labels)
    score = estimator.score(new_samples, labels)
    expect(score).to eq(1.0)
  end
end
