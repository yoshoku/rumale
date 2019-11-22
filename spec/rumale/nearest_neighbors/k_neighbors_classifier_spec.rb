# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::NearestNeighbors::KNeighborsClassifier do
  let(:three_clusters) { three_clusters_dataset }
  let(:x) { three_clusters[0] }
  let(:y) { three_clusters[1] }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_classes) { 3 }
  let(:estimator) { described_class.new(n_neighbors: 5).fit(x, y) }
  let(:predicted) { estimator.predict(x) }
  let(:score) { estimator.score(x, y) }
  let(:copied) { Marshal.load(Marshal.dump(estimator)) }

  it 'classifies three clusters data.', :aggregate_failures do
    expect(estimator.prototypes.class).to eq(Numo::DFloat)
    expect(estimator.prototypes.ndim).to eq(2)
    expect(estimator.prototypes.shape[0]).to eq(n_samples)
    expect(estimator.prototypes.shape[1]).to eq(n_features)
    expect(estimator.labels.class).to eq(Numo::Int32)
    expect(estimator.labels.ndim).to eq(1)
    expect(estimator.labels.shape[0]).to eq(n_samples)
    expect(estimator.classes.class).to eq(Numo::Int32)
    expect(estimator.classes.ndim).to eq(1)
    expect(estimator.classes.shape[0]).to eq(n_classes)
    expect(predicted.class).to eq(Numo::Int32)
    expect(predicted.ndim).to eq(1)
    expect(predicted.shape[0]).to eq(n_samples)
    expect(predicted).to eq(y)
    expect(score).to eq(1.0)
  end

  it 'dumps and restores itself using Marshal module.', :aggregate_failures do
    expect(estimator.class).to eq(copied.class)
    expect(estimator.params[:n_neighbors]).to eq(copied.params[:n_neighbors])
    expect(estimator.prototypes).to eq(copied.prototypes)
    expect(estimator.labels).to eq(copied.labels)
    expect(estimator.classes).to eq(copied.classes)
    expect(score).to eq(copied.score(x, y))
  end
end
