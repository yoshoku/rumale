# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Pipeline::Pipeline do
  let(:x) { Marshal.load(File.read(__dir__ + '/../test_samples_xor.dat')) }
  let(:y) { Marshal.load(File.read(__dir__ + '/../test_labels_xor.dat')) }
  let(:xc) { Marshal.load(File.read(__dir__ + '/../test_samples_three_clusters.dat')) }
  let(:yc) { Marshal.load(File.read(__dir__ + '/../test_labels_three_clusters.dat')) }
  let(:n_rbf_comps) { 64 }
  let(:n_nmf_comps) { 8 }
  let(:n_pca_comps) { 4 }
  let(:n_clusters) { 3 }
  let(:rbf) { Rumale::KernelApproximation::RBF.new(gamma: 0.1, n_components: n_rbf_comps, random_seed: 1) }
  let(:pca) { Rumale::Decomposition::PCA.new(n_components: n_pca_comps, tol: 1.0e-8, random_seed: 1) }
  let(:nmf) { Rumale::Decomposition::NMF.new(n_components: n_nmf_comps, random_seed: 1) }
  let(:svc) { Rumale::LinearModel::SVC.new(random_seed: 1) }
  let(:nrm) { Rumale::Preprocessing::L2Normalizer.new }
  let(:nbs) { Rumale::NaiveBayes::GaussianNB.new }
  let(:kms) { Rumale::Clustering::KMeans.new(n_clusters: n_clusters, random_seed: 1) }

  it 'classifies xor data with Kernel approximation, PCA, and SVC.' do
    n_samples, n_features = x.shape

    pipe = described_class.new(steps: { rbf: rbf, pca: pca, skip: nil, nrm: nrm, svc: svc })
    pipe.fit(x, y)

    expect(pipe.steps[:rbf].random_mat.shape).to match([n_features, n_rbf_comps])
    expect(pipe.steps[:pca].components.shape).to match([n_pca_comps, n_rbf_comps])
    expect(pipe.steps[:svc].weight_vec.shape).to match([n_pca_comps])
    expect(pipe.predict(x).shape).to match([n_samples])
    expect(pipe.decision_function(x).shape).to match([n_samples])
    expect(pipe.score(x, y)).to eq(1.0)
  end

  it 'analyzes clusters with Kernel approximation and K-Means.' do
    n_samples, = xc.shape

    pipe = described_class.new(steps: { rbf: rbf, kms: kms })
    predicted = pipe.fit_predict(xc)

    expect(predicted.shape).to match([n_samples])
    expect(pipe.steps[:kms].cluster_centers.shape).to match([n_clusters, n_rbf_comps])
    expect(pipe.score(xc, yc)).to eq(1.0)
  end

  it 'transforms high-dimensional data with NMF and PCA.' do
    n_samples, n_features = x.shape
    n_high_features = 16
    projected_x = x.abs.dot(Numo::DFloat.new(n_features, n_high_features).rand)

    pipe = described_class.new(steps: { nmf: nmf, pca: pca })
    trans_x = pipe.fit_transform(projected_x, y)
    expect(trans_x.shape).to match([n_samples, n_pca_comps])
    expect(pipe.steps[:nmf].components.shape).to match([n_nmf_comps, n_high_features])
    expect(pipe.steps[:pca].components.shape).to match([n_pca_comps, n_nmf_comps])

    trans_x = pipe.transform(projected_x)
    expect(trans_x.shape).to match([n_samples, n_pca_comps])

    reconst_x = pipe.inverse_transform(trans_x)
    expect(reconst_x.shape).to match([n_samples, n_high_features])
  end

  it 'estimates class probabilities with Kernel approximation, PCA, and Naive bayes.' do
    n_samples, = x.shape
    classes = y.to_a.uniq.sort

    pipe = described_class.new(steps: { rbf: rbf, pca: pca, nbs: nbs })
    pipe.fit(x, y)

    probs = pipe.predict_proba(x)
    predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[probs[n, true].max_index] })]
    expect(probs.shape).to match([n_samples, 2])
    expect(predicted).to eq(y)

    log_probs = pipe.predict_log_proba(x)
    predicted = Numo::Int32[*(Array.new(n_samples) { |n| classes[log_probs[n, true].max_index] })]
    expect(log_probs.shape).to match([n_samples, 2])
    expect(predicted).to eq(y)
  end

  it 'dumps and restores itself using Marshal module.' do
    pipe = described_class.new(steps: { rbf: rbf, pca: pca, svc: svc })
    pipe.fit(x, y)
    copied = Marshal.load(Marshal.dump(pipe))
    expect(copied.class).to eq(pipe.class)
    expect(copied.steps).to include(*pipe.steps.keys)
    expect(copied.steps[:rbf].random_mat).to eq(pipe.steps[:rbf].random_mat)
    expect(copied.steps[:pca].components).to eq(pipe.steps[:pca].components)
    expect(copied.steps[:svc].weight_vec).to eq(pipe.steps[:svc].weight_vec)
  end

  it 'raises TypeError when given steps including a non-transformer and non-estimator.' do
    expect { described_class.new(steps: { rbf: rbf, bad: 'skip', svc: svc }) }.to raise_error(TypeError)
    expect { described_class.new(steps: { rbf: rbf, bad: 'skip' }) }.to raise_error(TypeError)
  end
end
