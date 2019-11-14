# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NeuralNetwork::MLPClassifier do
  let(:x) { dataset[0] }
  let(:y) { dataset[1] }
  let(:classes) { y.to_a.uniq.sort }
  let(:n_samples) { x.shape[0] }
  let(:n_classes) { classes.size }
  let(:estimator) { described_class.new(hidden_units: [32, 16], max_iter: 100, verbose: false, random_seed: 1).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:probs) { estimator.predict_proba(x) }
  let(:score) { estimator.score(x, y) }
  let(:predicted_by_probs) { Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })] }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  shared_examples 'classification' do
    it 'classifies given dataset.', :aggregate_failures do
      expect(estimator.classes.class).to eq(Numo::Int32)
      expect(estimator.classes.size).to eq(n_classes)
      expect(estimator.classes.ndim).to eq(1)
      expect(estimator.classes.shape[0]).to eq(n_classes)
      expect(predicted.class).to eq(Numo::Int32)
      expect(predicted.ndim).to eq(1)
      expect(predicted.shape[0]).to eq(n_samples)
      expect(predicted).to eq(y)
      expect(probs.class).to eq(Numo::DFloat)
      expect(probs.ndim).to eq(2)
      expect(probs.shape[0]).to eq(n_samples)
      expect(probs.shape[1]).to eq(n_classes)
      expect(predicted_by_probs).to eq(y)
      expect(score).to eq(1.0)
    end
  end

  shared_examples 'dump and load model' do
    it 'dumps and restores itself using Marshal module.', :aggregate_failures do
      expect(estimator.class).to eq(copied.class)
      expect(estimator.params).to eq(copied.params)
      expect(score).to eq(copied.score(x, y))
    end
  end

  context 'when binary classification problem' do
    let(:dataset) { xor_dataset }

    it_behaves_like 'classification'
    it_behaves_like 'dump and load model'
  end

  context 'when multiclass classification problem' do
    let(:dataset) { three_clusters_dataset }

    it_behaves_like 'classification'
    it_behaves_like 'dump and load model'
  end
end
