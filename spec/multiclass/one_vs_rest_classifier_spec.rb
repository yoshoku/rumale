require 'spec_helper'

RSpec.describe SVMKit::Multiclass::OneVsRestClassifier do
  let(:samples) do
    SVMKit::Utils.restore_nmatrix(Marshal.load(File.read(__dir__ + '/test_samples_three_clusters.dat')))
  end
  let(:labels) do
    SVMKit::Utils.restore_nmatrix(Marshal.load(File.read(__dir__ + '/test_labels_three_clusters.dat')))
  end
  let(:base_estimator) do
    SVMKit::LinearModel::PegasosSVC.new(penalty: 1.0, max_iter: 100, batch_size: 20, random_seed: 1)
  end
  let(:estimator) { described_class.new(estimator: base_estimator) }

  it 'classifies three clusters.' do
    estimator.fit(samples, labels)
    score = estimator.score(samples, labels)
    expect(score).to eq(1.0)
  end

  it 'dumps and restores itself using Marshal module.' do
    estimator.fit(samples, labels)
    copied = Marshal.load(Marshal.dump(estimator))
    expect(estimator.class).to eq(copied.class)
    expect(estimator.estimators.size).to eq(copied.estimators.size)
    expect(estimator.estimators[0].class).to eq(copied.estimators[0].class)
    expect(estimator.estimators[1].class).to eq(copied.estimators[1].class)
    expect(estimator.estimators[2].class).to eq(copied.estimators[2].class)
    expect(estimator.estimators[0].weight_vec).to eq(copied.estimators[0].weight_vec)
    expect(estimator.estimators[1].weight_vec).to eq(copied.estimators[1].weight_vec)
    expect(estimator.estimators[2].weight_vec).to eq(copied.estimators[2].weight_vec)
    expect(estimator.classes).to eq(copied.classes)
    expect(estimator.params[:estimator].class).to eq(copied.params[:estimator].class)
  end
end
