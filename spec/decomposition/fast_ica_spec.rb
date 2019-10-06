# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Rumale::Decomposition::FastICA do
  let(:n_components) { 2 }
  let(:n_features) { 7 }
  let(:n_samples) { 1000 }
  let(:whiten) { true }
  let(:fun) { 'logcosh' }
  let(:sources) do
    t = Numo::DFloat.linspace(0, 100, n_samples)
    s1 = Numo::NMath.cos(Math::PI * t)
    s2 = Numo::NMath.sin(Math::PI * t + Math::PI / 4).abs
    s = Numo::NArray.vstack([s1, s2]).transpose.dup
    s -= s.mean(0)
    s / s.var(0)
  end
  let(:mixing_mat) { Rumale::Utils.rand_normal([n_components, n_features], Random.new(1)) }
  let(:observed) { sources.dot(mixing_mat) }
  let(:analyzer) { described_class.new(n_components: n_components, whiten: whiten, fun: fun, tol: 1e-8, random_seed: 1) }
  let(:reconstructed) { analyzer.fit_transform(observed) }
  let(:inv_error) { Math.sqrt(((observed - analyzer.inverse_transform(reconstructed))**2).sum) / n_samples }
  # Since the reconstructed signals with ICA are arbitrary in scale and order,
  # normalization is necessary fo comparison to the source signals.
  let(:permutated) do
    s1 = sources[true, 0]
    s2 = sources[true, 1]
    r1 = reconstructed[true, 0]
    r2 = reconstructed[true, 1]
    r1, r2 = [r2, r1] if r1.dot(s1).abs < r1.dot(s2).abs
    r1 *= r1.dot(s1).positive? ? 1 : -1
    r2 *= r2.dot(s2).positive? ? 1 : -1
    Numo::NArray.vstack([r1, r2]).transpose.dup
  end
  let(:rec_error) do
    diff = (sources / sources[0, true] - permutated / permutated[0, true])
    Math.sqrt((diff**2).sum) / n_samples
  end

  context 'when the contrast function is logcosh' do
    it 'reconstructs the source signals', aggregate_failures: true do
      expect(reconstructed.class).to eq(Numo::DFloat)
      expect(reconstructed.shape[0]).to eq(n_samples)
      expect(reconstructed.shape[1]).to eq(n_components)
      expect(analyzer.n_iter).to be > 1
      expect(analyzer.components.class).to eq(Numo::DFloat)
      expect(analyzer.components.shape[0]).to eq(n_components)
      expect(analyzer.components.shape[1]).to eq(n_features)
      expect(analyzer.mixing.class).to eq(Numo::DFloat)
      expect(analyzer.mixing.shape[0]).to eq(n_features)
      expect(analyzer.mixing.shape[1]).to eq(n_components)
      expect(rec_error).to be < 1e-3
    end

    it 'inverse transforms' do
      expect(inv_error).to be < 1e-3
    end

    it 'dumps and restores itself using Marshal module', aggregate_failures: true do
      copied = Marshal.load(Marshal.dump(analyzer.fit(observed)))
      err = Math.sqrt(((analyzer.transform(observed) - copied.transform(observed))**2).sum) / n_samples
      expect(analyzer.class).to eq(copied.class)
      expect(analyzer.params[:n_components]).to eq(copied.params[:n_components])
      expect(analyzer.params[:whiten]).to eq(copied.params[:whiten])
      expect(analyzer.params[:fun]).to eq(copied.params[:fun])
      expect(analyzer.params[:alpha]).to eq(copied.params[:alpha])
      expect(analyzer.params[:max_iter]).to eq(copied.params[:max_iter])
      expect(analyzer.params[:tol]).to eq(copied.params[:tol])
      expect(analyzer.params[:random_seed]).to eq(copied.params[:random_seed])
      expect(analyzer.components).to eq(copied.components)
      expect(analyzer.mixing).to eq(copied.mixing)
      expect(analyzer.n_iter).to eq(copied.n_iter)
      expect(analyzer.rng).to eq(copied.rng)
      expect(err).to be < 1e-8
    end
  end

  context 'when the contrast function is exp' do
    let(:fun) { 'exp' }

    it { expect(rec_error).to be < 1e-3 }
  end

  context 'when the contrast function is cube' do
    let(:fun) { 'cube' }

    it { expect(rec_error).to be < 1e-3 }
  end

  context 'when the number of components is one' do
    let(:n_components) { 1 }
    let(:analyzer) { described_class.new(n_components: n_components, random_seed: 1) }
    let(:reconstructed) { analyzer.fit_transform(Rumale::Utils.rand_uniform([n_samples, n_features], Random.new(1))) }
    let(:inversed) { analyzer.inverse_transform(reconstructed.expand_dims(1)) }

    it 'analyzes one independent component', aggregate_failures: true do
      expect(reconstructed.class).to eq(Numo::DFloat)
      expect(reconstructed.shape[0]).to eq(n_samples)
      expect(reconstructed.shape[1]).to be_nil
      expect(inversed.class).to eq(Numo::DFloat)
      expect(inversed.shape[0]).to eq(n_samples)
      expect(inversed.shape[1]).to eq(n_features)
      expect(analyzer.components.shape[0]).to eq(n_features)
      expect(analyzer.components.shape[1]).to be_nil
      expect(analyzer.mixing.shape[0]).to eq(n_features)
      expect(analyzer.mixing.shape[1]).to be_nil
    end
  end

  context 'when the given whiten data' do
    let(:whiten) { false }
    let(:observed) do
      x = sources.dot(mixing_mat)
      mean, whiten_mat = analyzer.send(:whitening, x, n_components)
      (x - mean).dot(whiten_mat.transpose)
    end

    it 'reconstructs the source signals', aggregate_failures: true do
      expect(reconstructed.class).to eq(Numo::DFloat)
      expect(reconstructed.shape[0]).to eq(n_samples)
      expect(reconstructed.shape[1]).to eq(n_components)
      expect(analyzer.components.class).to eq(Numo::DFloat)
      expect(analyzer.components.shape[0]).to eq(n_components)
      expect(analyzer.components.shape[1]).to eq(n_components)
      expect(analyzer.mixing.class).to eq(Numo::DFloat)
      expect(analyzer.mixing.shape[0]).to eq(n_components)
      expect(analyzer.mixing.shape[1]).to eq(n_components)
      expect(rec_error).to be < 1e-3
      expect(inv_error).to be < 1e-3
    end
  end
end
