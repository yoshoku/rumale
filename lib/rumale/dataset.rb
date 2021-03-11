# frozen_string_literal: true

require 'csv'
require 'rumale/validation'
require 'rumale/utils'
require 'rumale/preprocessing/min_max_scaler'

module Rumale
  # Module for loading and saving a dataset file.
  module Dataset
    class << self
      # Load a dataset with the libsvm file format into Numo::NArray.
      #
      # @param filename [String] A path to a dataset file.
      # @param n_features [Integer/Nil] The number of features of data to load.
      #   If nil is given, it will be detected automatically from given file.
      # @param zero_based [Boolean] Whether the column index starts from 0 (true) or 1 (false).
      # @param dtype [Numo::NArray] Data type of Numo::NArray for features to be loaded.
      #
      # @return [Array<Numo::NArray>]
      #   Returns array containing the (n_samples x n_features) matrix for feature vectors
      #   and (n_samples) vector for labels or target values.
      def load_libsvm_file(filename, n_features: nil, zero_based: false, dtype: Numo::DFloat)
        ftvecs = []
        labels = []
        n_features_detected = 0
        CSV.foreach(filename, col_sep: "\s", headers: false) do |line|
          label, ftvec, max_idx = parse_libsvm_line(line, zero_based)
          labels.push(label)
          ftvecs.push(ftvec)
          n_features_detected = max_idx if n_features_detected < max_idx
        end
        n_features ||= n_features_detected
        n_features = [n_features, n_features_detected].max
        [convert_to_matrix(ftvecs, n_features, dtype), Numo::NArray.asarray(labels)]
      end

      # Dump the dataset with the libsvm file format.
      #
      # @param data [Numo::NArray] (shape: [n_samples, n_features]) matrix consisting of feature vectors.
      # @param labels [Numo::NArray] (shape: [n_samples]) matrix consisting of labels or target values.
      # @param filename [String] A path to the output libsvm file.
      # @param zero_based [Boolean] Whether the column index starts from 0 (true) or 1 (false).
      def dump_libsvm_file(data, labels, filename, zero_based: false)
        n_samples = [data.shape[0], labels.shape[0]].min
        single_label = labels.shape[1].nil?
        label_type = detect_dtype(labels)
        value_type = detect_dtype(data)
        File.open(filename, 'w') do |file|
          n_samples.times do |n|
            label = single_label ? labels[n] : labels[n, true].to_a
            file.puts(dump_libsvm_line(label, data[n, true],
                                       label_type, value_type, zero_based))
          end
        end
      end

      # Generate a two-dimensional data set consisting of an inner circle and an outer circle.
      #
      # @param n_samples [Integer] The number of samples.
      # @param shuffle [Boolean] The flag indicating whether to shuffle the dataset
      # @param noise [Float] The standard deviaion of gaussian noise added to the data.
      #   If nil is given, no noise is added.
      # @param factor [Float] The scale factor between inner and outer circles. The interval of factor is (0, 1).
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def make_circles(n_samples, shuffle: true, noise: nil, factor: 0.8, random_seed: nil)
        Rumale::Validation.check_params_numeric(n_samples: n_samples, factor: factor)
        Rumale::Validation.check_params_boolean(shuffle: shuffle)
        Rumale::Validation.check_params_numeric_or_nil(noise: noise, random_seed: random_seed)
        raise ArgumentError, 'The number of samples must be more than 2.' if n_samples <= 1
        raise RangeError, 'The interval of factor is (0, 1).' if factor <= 0 || factor >= 1

        # initialize some variables.
        rs = random_seed
        rs ||= srand
        rng = Random.new(rs)
        n_samples_out = n_samples.fdiv(2).to_i
        n_samples_in = n_samples - n_samples_out
        # make two circles.
        linsp_out = Numo::DFloat.linspace(0, 2 * Math::PI, n_samples_out)
        linsp_in = Numo::DFloat.linspace(0, 2 * Math::PI, n_samples_in)
        circle_out = Numo::DFloat[Numo::NMath.cos(linsp_out), Numo::NMath.sin(linsp_out)].transpose
        circle_in = Numo::DFloat[Numo::NMath.cos(linsp_in), Numo::NMath.sin(linsp_in)].transpose
        x = Numo::DFloat.vstack([circle_out, factor * circle_in])
        y = Numo::Int32.hstack([Numo::Int32.zeros(n_samples_out), Numo::Int32.ones(n_samples_in)])
        # shuffle data indices.
        if shuffle
          rand_ids = Array(0...n_samples).shuffle(random: rng.dup)
          x = x[rand_ids, true].dup
          y = y[rand_ids].dup
        end
        # add gaussian noise.
        x += Rumale::Utils.rand_normal(x.shape, rng.dup, 0.0, noise) unless noise.nil?
        [x, y]
      end

      # Generate a two-dimensional data set consisting of two half circles shifted.
      #
      # @param n_samples [Integer] The number of samples.
      # @param shuffle [Boolean] The flag indicating whether to shuffle the dataset
      # @param noise [Float] The standard deviaion of gaussian noise added to the data.
      #   If nil is given, no noise is added.
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def make_moons(n_samples, shuffle: true, noise: nil, random_seed: nil)
        Rumale::Validation.check_params_numeric(n_samples: n_samples)
        Rumale::Validation.check_params_boolean(shuffle: shuffle)
        Rumale::Validation.check_params_numeric_or_nil(noise: noise, random_seed: random_seed)
        raise ArgumentError, 'The number of samples must be more than 2.' if n_samples <= 1

        # initialize some variables.
        rs = random_seed
        rs ||= srand
        rng = Random.new(rs)
        n_samples_out = n_samples.fdiv(2).to_i
        n_samples_in = n_samples - n_samples_out
        # make two half circles.
        linsp_out = Numo::DFloat.linspace(0, Math::PI, n_samples_out)
        linsp_in = Numo::DFloat.linspace(0, Math::PI, n_samples_in)
        circle_out = Numo::DFloat[Numo::NMath.cos(linsp_out), Numo::NMath.sin(linsp_out)].transpose
        circle_in = Numo::DFloat[1 - Numo::NMath.cos(linsp_in), 1 - Numo::NMath.sin(linsp_in) - 0.5].transpose
        x = Numo::DFloat.vstack([circle_out, circle_in])
        y = Numo::Int32.hstack([Numo::Int32.zeros(n_samples_out), Numo::Int32.ones(n_samples_in)])
        # shuffle data indices.
        if shuffle
          rand_ids = Array(0...n_samples).shuffle(random: rng.dup)
          x = x[rand_ids, true].dup
          y = y[rand_ids].dup
        end
        # add gaussian noise.
        x += Rumale::Utils.rand_normal(x.shape, rng.dup, 0.0, noise) unless noise.nil?
        [x, y]
      end

      # Generate Gaussian blobs.
      #
      # @param n_samples [Integer] The total number of samples.
      # @param n_features [Integer] The number of features.
      #   If "centers" parameter is given as a Numo::DFloat array, this parameter is ignored.
      # @param centers [Integer/Numo::DFloat/Nil] The number of cluster centroids or the fixed cluster centroids.
      #   If nil is given, the number of cluster centroids is set to 3.
      # @param cluster_std [Float] The standard deviation of the clusters.
      # @param center_box [Array] The bounding box for each cluster centroids.
      #   If "centers" parameter is given as a Numo::DFloat array, this parameter is ignored.
      # @param shuffle [Boolean] The flag indicating whether to shuffle the dataset
      # @param random_seed [Integer] The seed value using to initialize the random generator.
      def make_blobs(n_samples = 1000, n_features = 2,
                     centers: nil, cluster_std: 1.0, center_box: [-10, 10], shuffle: true, random_seed: nil)
        Rumale::Validation.check_params_numeric(n_samples: n_samples, n_features: n_features, cluster_std: cluster_std)
        Rumale::Validation.check_params_type(Array, center_box: center_box)
        Rumale::Validation.check_params_boolean(shuffle: shuffle)
        Rumale::Validation.check_params_numeric_or_nil(random_seed: random_seed)
        # initialize rng.
        rs = random_seed
        rs ||= srand
        rng = Random.new(rs)
        # initialize centers.
        if centers.is_a?(Numo::DFloat)
          n_centers = centers.shape[0]
          n_features = centers.shape[1]
        else
          n_centers = centers.is_a?(Integer) ? centers : 3
          center_min = center_box.first
          center_max = center_box.last
          centers = Rumale::Utils.rand_uniform([n_centers, n_features], rng)
          normalizer = Rumale::Preprocessing::MinMaxScaler.new(feature_range: [center_min, center_max])
          centers = normalizer.fit_transform(centers)
        end
        # generate blobs.
        sz_cluster = [n_samples / n_centers] * n_centers
        (n_samples % n_centers).times { |n| sz_cluster[n] += 1 }
        x = Rumale::Utils.rand_normal([sz_cluster[0], n_features], rng, 0.0, cluster_std) + centers[0, true]
        y = Numo::Int32.zeros(sz_cluster[0])
        (1...n_centers).each do |n|
          c = Rumale::Utils.rand_normal([sz_cluster[n], n_features], rng, 0.0, cluster_std) + centers[n, true]
          x = Numo::DFloat.vstack([x, c])
          y = y.concatenate(Numo::Int32.zeros(sz_cluster[n]) + n)
        end
        # shuffle data.
        if shuffle
          rand_ids = Array(0...n_samples).shuffle(random: rng.dup)
          x = x[rand_ids, true].dup
          y = y[rand_ids].dup
        end
        [x, y]
      end

      private

      def parse_libsvm_line(line, zero_based)
        label = parse_label(line.shift)
        adj_idx = zero_based == false ? 1 : 0
        max_idx = -1
        ftvec = []
        while (el = line.shift)
          idx, val = el.split(':')
          idx = idx.to_i - adj_idx
          val = val.to_i.to_s == val ? val.to_i : val.to_f
          max_idx = idx if max_idx < idx
          ftvec.push([idx, val])
        end
        [label, ftvec, max_idx]
      end

      def parse_label(label)
        lbl_arr = label.split(',').map { |lbl| lbl.to_i.to_s == lbl ? lbl.to_i : lbl.to_f }
        lbl_arr.size > 1 ? lbl_arr : lbl_arr[0]
      end

      def convert_to_matrix(data, n_features, dtype)
        mat = []
        data.each do |ft|
          vec = Array.new(n_features) { 0 }
          ft.each { |el| vec[el[0]] = el[1] }
          mat.push(vec)
        end
        dtype.asarray(mat)
      end

      def detect_dtype(data)
        arr_type_str = Numo::NArray.array_type(data).to_s
        type = '%s'
        type = '%d' if ['Numo::Int8', 'Numo::Int16', 'Numo::Int32', 'Numo::Int64'].include?(arr_type_str)
        type = '%d' if ['Numo::UInt8', 'Numo::UInt16', 'Numo::UInt32', 'Numo::UInt64'].include?(arr_type_str)
        type = '%.10g' if ['Numo::SFloat', 'Numo::DFloat'].include?(arr_type_str)
        type
      end

      def dump_libsvm_line(label, ftvec, label_type, value_type, zero_based)
        line = dump_label(label, label_type.to_s)
        ftvec.to_a.each_with_index do |val, n|
          idx = n + (zero_based == false ? 1 : 0)
          line += format(" %d:#{value_type}", idx, val) if val != 0
        end
        line
      end

      def dump_label(label, label_type_str)
        if label.is_a?(Array)
          label.map { |lbl| format(label_type_str, lbl) }.join(',')
        else
          format(label_type_str, label)
        end
      end
    end
  end
end
