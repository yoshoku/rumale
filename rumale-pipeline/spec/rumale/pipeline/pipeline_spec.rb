# frozen_string_literal: true

require 'spec_helper'

require 'rumale/kernel_approximation/rbf'
require 'rumale/decomposition/pca'
require 'rumale/decomposition/nmf'
require 'rumale/linear_model/svc'
require 'rumale/preprocessing/l2_normalizer'
require 'rumale/naive_bayes/gaussian_nb'
require 'rumale/clustering/k_means'

RSpec.describe Rumale::Pipeline::Pipeline do
  let(:xor) { xor_dataset }
  let(:three_clusters) { three_clusters_dataset }
  let(:n_samples) { x.shape[0] }
  let(:n_features) { x.shape[1] }
  let(:n_rbf_comps) { 64 }
  let(:n_nmf_comps) { 8 }
  let(:n_pca_comps) { 4 }
  let(:n_clusters) { 3 }
  let(:rbf) { Rumale::KernelApproximation::RBF.new(gamma: 0.1, n_components: n_rbf_comps, random_seed: 1) }
  let(:pca) { Rumale::Decomposition::PCA.new(n_components: n_pca_comps, tol: 1.0e-8, random_seed: 1) }
  let(:nmf) { Rumale::Decomposition::NMF.new(n_components: n_nmf_comps, random_seed: 1) }
  let(:svc) { Rumale::LinearModel::SVC.new }
  let(:nrm) { Rumale::Preprocessing::L2Normalizer.new }
  let(:nbs) { Rumale::NaiveBayes::GaussianNB.new }
  let(:kms) { Rumale::Clustering::KMeans.new(n_clusters: n_clusters, random_seed: 1) }
  let(:score) { pipe.score(x, y) }

  context 'when classification task' do
    let(:x) { xor[0] }
    let(:y) { xor[1] }
    let(:pipe) { described_class.new(steps: { rbf: rbf, pca: pca, skip: nil, nrm: nrm, svc: svc }).fit(x, y) }

    it 'classifies xor data with Kernel approximation, PCA, and SVC.', :aggregate_failures do
      expect(pipe.steps[:rbf].random_mat.shape).to match([n_features, n_rbf_comps])
      expect(pipe.steps[:pca].components.shape).to match([n_pca_comps, n_rbf_comps])
      expect(pipe.steps[:svc].weight_vec.shape).to match([n_pca_comps])
      expect(pipe.predict(x).shape).to match([n_samples])
      expect(pipe.decision_function(x).shape).to match([n_samples])
      expect(score).to eq(1.0)
    end

    context 'when calculating class probabilities', :aggregate_failures do
      let(:classes) { y.to_a.uniq.sort }
      let(:pipe) { described_class.new(steps: { rbf: rbf, pca: pca, nbs: nbs }).fit(x, y) }
      let(:probs) { pipe.predict_proba(x) }
      let(:log_probs) { pipe.predict_log_proba(x) }
      let(:predicted_by_probs) { Numo::Int32[*Array.new(n_samples) { |n| classes[probs[n, true].max_index] }] }
      let(:predicted_by_log_probs) { Numo::Int32[*Array.new(n_samples) { |n| classes[log_probs[n, true].max_index] }] }

      it 'estimates class probabilities with Kernel approximation, PCA, and Naive bayes.', :aggregate_failures do
        expect(probs.shape).to match([n_samples, 2])
        expect(predicted_by_probs).to eq(y)
        expect(log_probs.shape).to match([n_samples, 2])
        expect(predicted_by_log_probs).to eq(y)
      end
    end
  end

  context 'when clustering task' do
    let(:x) { three_clusters[0] }
    let(:y) { three_clusters[1] }
    let(:pipe) { described_class.new(steps: { rbf: rbf, kms: kms }) }
    let(:predicted) { pipe.fit_predict(x) }

    it 'analyzes clusters with Kernel approximation and K-Means.', :aggregate_failures do
      expect(predicted.shape).to match([n_samples])
      expect(pipe.steps[:kms].cluster_centers.shape).to match([n_clusters, n_rbf_comps])
      expect(score).to eq(1.0)
    end
  end

  context 'when dimensionality reduction task' do
    let(:x) { xor[0] }
    let(:y) { xor[1] }
    let(:n_high_features) { 16 }
    let(:projected_x) { x.abs.dot(Numo::DFloat.new(n_features, n_high_features).rand) }
    let(:pipe) { described_class.new(steps: { nmf: nmf, pca: pca }) }
    let(:z_fit_transform) { pipe.fit_transform(projected_x, y) }
    let(:z_fit_and_transform) { pipe.fit(projected_x, y).transform(projected_x) }
    let(:reconst_x) { pipe.inverse_transform(z_fit_transform) }

    it 'transforms high-dimensional data with NMF and PCA.', :aggregate_failures do
      expect(z_fit_transform.shape).to match([n_samples, n_pca_comps])
      expect(z_fit_and_transform.shape).to match([n_samples, n_pca_comps])
      expect(pipe.steps[:nmf].components.shape).to match([n_nmf_comps, n_high_features])
      expect(pipe.steps[:pca].components.shape).to match([n_pca_comps, n_nmf_comps])
      expect(reconst_x.shape).to match([n_samples, n_high_features])
    end
  end

  it 'raises TypeError when given steps including a non-transformer and non-estimator.', :aggregate_failures do
    expect { described_class.new(steps: { rbf: rbf, bad: 'skip', svc: svc }) }.to raise_error(TypeError)
    expect { described_class.new(steps: { rbf: rbf, bad: 'skip' }) }.to raise_error(TypeError)
  end
end
