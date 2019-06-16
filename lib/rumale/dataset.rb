# frozen_string_literal: true

require 'csv'
require 'rumale/validation'

module Rumale
  # Module for loading and saving a dataset file.
  module Dataset
    class << self
      # Load a dataset with the libsvm file format into Numo::NArray.
      #
      # @param filename [String] A path to a dataset file.
      # @param zero_based [Boolean] Whether the column index starts from 0 (true) or 1 (false).
      # @param dtype [Numo::NArray] Data type of Numo::NArray for features to be loaded.
      #
      # @return [Array<Numo::NArray>]
      #   Returns array containing the (n_samples x n_features) matrix for feature vectors
      #   and (n_samples) vector for labels or target values.
      def load_libsvm_file(filename, zero_based: false, dtype: Numo::DFloat)
        ftvecs = []
        labels = []
        n_features = 0
        CSV.foreach(filename, col_sep: "\s", headers: false) do |line|
          label, ftvec, max_idx = parse_libsvm_line(line, zero_based)
          labels.push(label)
          ftvecs.push(ftvec)
          n_features = max_idx if n_features < max_idx
        end
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
      # @random_seed [Integer] The seed value using to initialize the random generator.
      def make_circles(n_samples, shuffle: true, noise: nil, factor: 0.8, random_seed: nil)
        Rumale::Validation.check_params_integer(n_samples: n_samples)
        Rumale::Validation.check_params_boolean(shuffle: shuffle)
        Rumale::Validation.check_params_type_or_nil(Float, noise: noise)
        Rumale::Validation.check_params_float(factor: factor)
        Rumale::Validation.check_params_type_or_nil(Integer, random_seed: random_seed)
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
          rand_ids = [*0...n_samples].shuffle(random: rng.dup)
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
      # @random_seed [Integer] The seed value using to initialize the random generator.
      def make_moons(n_samples, shuffle: true, noise: nil, random_seed: nil)
        Rumale::Validation.check_params_integer(n_samples: n_samples)
        Rumale::Validation.check_params_boolean(shuffle: shuffle)
        Rumale::Validation.check_params_type_or_nil(Float, noise: noise)
        Rumale::Validation.check_params_type_or_nil(Integer, random_seed: random_seed)
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
          rand_ids = [*0...n_samples].shuffle(random: rng.dup)
          x = x[rand_ids, true].dup
          y = y[rand_ids].dup
        end
        # add gaussian noise.
        x += Rumale::Utils.rand_normal(x.shape, rng.dup, 0.0, noise) unless noise.nil?
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
          line += format(" %d:#{value_type}", idx, val) if val != 0.0
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
