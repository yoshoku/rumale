#include "rumale.h"

VALUE
create_zero_vector(const long n_dimensions)
{
  long i;
  VALUE vec = rb_ary_new2(n_dimensions);

  for (i = 0; i < n_dimensions; i++) {
    rb_ary_store(vec, i, DBL2NUM(0));
  }

  return vec;
}

double
calc_gini_coef(VALUE histogram, const long n_elements)
{
  long i;
  double el;
  double gini = 0.0;
  const long n_classes = RARRAY_LEN(histogram);

  for (i = 0; i < n_classes; i++) {
    el = NUM2DBL(rb_ary_entry(histogram, i)) / n_elements;
    gini += el * el;
  }

  return 1.0 - gini;
}

double
calc_entropy(VALUE histogram, const long n_elements)
{
  long i;
  double el;
  double entropy = 0.0;
  const long n_classes = RARRAY_LEN(histogram);

  for (i = 0; i < n_classes; i++) {
    el = NUM2DBL(rb_ary_entry(histogram, i)) / n_elements;
    entropy += el * log(el + 1.0);
  }

  return -entropy;
}

VALUE
calc_mean_vec(VALUE sum_vec, const long n_elements)
{
  long i;
  const long n_dimensions = RARRAY_LEN(sum_vec);
  VALUE mean_vec = rb_ary_new2(n_dimensions);

  for (i = 0; i < n_dimensions; i++) {
    rb_ary_store(mean_vec, i, DBL2NUM(NUM2DBL(rb_ary_entry(sum_vec, i)) / n_elements));
  }

  return mean_vec;
}

double
calc_vec_mae(VALUE vec_a, VALUE vec_b)
{
  long i;
  const long n_dimensions = RARRAY_LEN(vec_a);
  double sum = 0.0;
  double diff;

  for (i = 0; i < n_dimensions; i++) {
    diff = NUM2DBL(rb_ary_entry(vec_a, i)) - NUM2DBL(rb_ary_entry(vec_b, i));
    sum += fabs(diff);
  }

  return sum / n_dimensions;
}

double
calc_vec_mse(VALUE vec_a, VALUE vec_b)
{
  long i;
  const long n_dimensions = RARRAY_LEN(vec_a);
  double sum = 0.0;
  double diff;

  for (i = 0; i < n_dimensions; i++) {
    diff = NUM2DBL(rb_ary_entry(vec_a, i)) - NUM2DBL(rb_ary_entry(vec_b, i));
    sum += diff * diff;
  }

  return sum / n_dimensions;
}

double
calc_mae(VALUE target_vecs, VALUE sum_vec)
{
  long i;
  const long n_elements = RARRAY_LEN(target_vecs);
  double sum = 0.0;
  VALUE mean_vec = calc_mean_vec(sum_vec, n_elements);

  for (i = 0; i < n_elements; i++) {
    sum += calc_vec_mae(rb_ary_entry(target_vecs, i), mean_vec);
  }

  return sum / n_elements;
}

double
calc_mse(VALUE target_vecs, VALUE sum_vec)
{
  long i;
  const long n_elements = RARRAY_LEN(target_vecs);
  double sum = 0.0;
  VALUE mean_vec = calc_mean_vec(sum_vec, n_elements);

  for (i = 0; i < n_elements; i++) {
    sum += calc_vec_mse(rb_ary_entry(target_vecs, i), mean_vec);
  }

  return sum / n_elements;
}

double
calc_impurity_cls(VALUE criterion, VALUE histogram, const long n_elements)
{
  if (strcmp(StringValuePtr(criterion), "entropy") == 0) {
    return calc_entropy(histogram, n_elements);
  }
  return calc_gini_coef(histogram, n_elements);
}

double
calc_impurity_reg(VALUE criterion, VALUE target_vecs, VALUE sum_vec)
{
  if (strcmp(StringValuePtr(criterion), "mae") == 0) {
    return calc_mae(target_vecs, sum_vec);
  }
  return calc_mse(target_vecs, sum_vec);
}

void
increment_histogram(VALUE histogram, const long bin_id)
{
  const double updated = NUM2DBL(rb_ary_entry(histogram, bin_id)) + 1;
  rb_ary_store(histogram, bin_id, DBL2NUM(updated));
}

void
decrement_histogram(VALUE histogram, const long bin_id)
{
  const double updated = NUM2DBL(rb_ary_entry(histogram, bin_id)) - 1;
  rb_ary_store(histogram, bin_id, DBL2NUM(updated));
}

void
add_sum_vec(VALUE sum_vec, VALUE target)
{
  long i;
  const long n_dimensions = RARRAY_LEN(sum_vec);
  double el;

  for (i = 0; i < n_dimensions; i++) {
    el = NUM2DBL(rb_ary_entry(sum_vec, i)) + NUM2DBL(rb_ary_entry(target, i));
    rb_ary_store(sum_vec, i, DBL2NUM(el));
  }
}

void
sub_sum_vec(VALUE sum_vec, VALUE target)
{
  long i;
  const long n_dimensions = RARRAY_LEN(sum_vec);
  double el;

  for (i = 0; i < n_dimensions; i++) {
    el = NUM2DBL(rb_ary_entry(sum_vec, i)) - NUM2DBL(rb_ary_entry(target, i));
    rb_ary_store(sum_vec, i, DBL2NUM(el));
  }
}

/**
 * @!visibility private
 * Find for split point with maximum information gain.
 *
 * @overload find_split_params(criterion, impurity, sorted_features, sorted_labels, uniqed_features, n_classes) -> Array<Float>
 *
 * @param criterion [String] The function to evaluate spliting point. Supported criteria are 'gini' and 'entropy'.
 * @param impurity [Float] The impurity of whole dataset.
 * @param sorted_features [Numo::DFloat] (shape: [n_samples]) The feature values sorted in ascending order.
 * @param sorted_labels [Numo::Int32] (shape: [n_labels]) The labels sorted according to feature values.
 * @param uniqed_features [Numo::DFloat] (shape: [n_uniqed_features]) The unique feature values.
 * @param n_classes [Integer] The number of classes.
 * @return [Float] The array consists of optimal parameters including impurities of child nodes, threshold, and gain.
 */
static VALUE
find_split_params_cls(VALUE self, VALUE criterion, VALUE whole_impurity, VALUE sorted_f, VALUE sorted_y, VALUE uniqed_f, VALUE n_classes_)
{
  long i;
  long curr_pos;
  long next_pos;
  long n_l_elements;
  long n_r_elements;
  const long n_classes = NUM2LONG(n_classes_);
  const long n_elements = RARRAY_LEN(sorted_f);
  const long n_uniq_elements = RARRAY_LEN(uniqed_f);
  const double w_impurity = NUM2DBL(whole_impurity);
  double l_impurity;
  double r_impurity;
  double gain;
  double curr_el;
  double next_el;
  VALUE l_histogram = create_zero_vector(n_classes);
  VALUE r_histogram = create_zero_vector(n_classes);
  VALUE opt_params = rb_ary_new2(4);

  /* Initialize optimal parameters. */
  rb_ary_store(opt_params, 0, DBL2NUM(0));                /* left impurity */
  rb_ary_store(opt_params, 1, DBL2NUM(w_impurity));       /* right impurity */
  rb_ary_store(opt_params, 2, rb_ary_entry(uniqed_f, 0)); /* threshold */
  rb_ary_store(opt_params, 3, DBL2NUM(0));                /* gain */

  /* Initialize child node variables. */
  n_l_elements = 0;
  n_r_elements = n_elements;
  for (i = 0; i < n_elements; i++) {
    increment_histogram(r_histogram, NUM2LONG(rb_ary_entry(sorted_y, i)));
  }

  /* Find optimal parameters. */
  for (curr_pos = 0, next_pos = 0; curr_pos < n_uniq_elements - 1; curr_pos++) {
    /* Find new split point. */
    curr_el = NUM2DBL(rb_ary_entry(uniqed_f, curr_pos));
    next_el = NUM2DBL(rb_ary_entry(sorted_f, next_pos));
    while (next_pos < n_elements && next_el <= curr_el) {
      increment_histogram(l_histogram, NUM2LONG(rb_ary_entry(sorted_y, next_pos)));
      n_l_elements++;
      decrement_histogram(r_histogram, NUM2LONG(rb_ary_entry(sorted_y, next_pos)));
      n_r_elements--;
      next_el = NUM2DBL(rb_ary_entry(sorted_f, ++next_pos));
    }
    /* Calculate gain of new split. */
    l_impurity = calc_impurity_cls(criterion, l_histogram, n_l_elements);
    r_impurity = calc_impurity_cls(criterion, r_histogram, n_r_elements);
    gain = w_impurity - (n_l_elements * l_impurity + n_r_elements * r_impurity) / n_elements;
    /* Update optimal parameters. */
    if (gain > NUM2DBL(rb_ary_entry(opt_params, 3))) {
      rb_ary_store(opt_params, 0, DBL2NUM(l_impurity));
      rb_ary_store(opt_params, 1, DBL2NUM(r_impurity));
      rb_ary_store(opt_params, 2, DBL2NUM(0.5 * (curr_el + next_el)));
      rb_ary_store(opt_params, 3, DBL2NUM(gain));
    }
  }

  return opt_params;
}

/**
 * @!visibility private
 * Find for split point with maximum information gain.
 *
 * @overload find_split_params(criterion, impurity, sorted_features, sorted_targets, uniqed_features) -> Array<Float>
 *
 * @param criterion [String] The function to evaluate spliting point. Supported criteria are 'mae' and 'mse'.
 * @param impurity [Float] The impurity of whole dataset.
 * @param sorted_features [Numo::DFloat] (shape: [n_samples]) The feature values sorted in ascending order.
 * @param sorted_targets [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values sorted according to feature values.
 * @param uniqed_features [Numo::DFloat] (shape: [n_uniqed_features]) The unique feature values.
 * @return [Float] The array consists of optimal parameters including impurities of child nodes, threshold, and gain.
 */
static VALUE
find_split_params_reg(VALUE self, VALUE criterion, VALUE whole_impurity, VALUE sorted_f, VALUE sorted_y, VALUE uniqed_f)
{
  long i;
  long curr_pos;
  long next_pos;
  long n_l_elements;
  long n_r_elements;
  const long n_elements = RARRAY_LEN(sorted_f);
  const long n_uniq_elements = RARRAY_LEN(uniqed_f);
  const long n_dimensions = RARRAY_LEN(rb_ary_entry(sorted_y, 0));
  const double w_impurity = NUM2DBL(whole_impurity);
  double l_impurity;
  double r_impurity;
  double gain;
  double curr_el;
  double next_el;
  VALUE l_sum_vec = create_zero_vector(n_dimensions);
  VALUE r_sum_vec = create_zero_vector(n_dimensions);
  VALUE l_target_vecs = rb_ary_new();
  VALUE r_target_vecs = rb_ary_new();
  VALUE target;
  VALUE opt_params = rb_ary_new2(4);

  /* Initialize optimal parameters. */
  rb_ary_store(opt_params, 0, DBL2NUM(0));                /* left impurity */
  rb_ary_store(opt_params, 1, DBL2NUM(w_impurity));       /* right impurity */
  rb_ary_store(opt_params, 2, rb_ary_entry(uniqed_f, 0)); /* threshold */
  rb_ary_store(opt_params, 3, DBL2NUM(0));                /* gain */

  /* Initialize child node variables. */
  n_l_elements = 0;
  n_r_elements = n_elements;
  for (i = 0; i < n_elements; i++) {
    target = rb_ary_entry(sorted_y, i);
    add_sum_vec(r_sum_vec, target);
    rb_ary_push(r_target_vecs, target);
  }

  /* Find optimal parameters. */
  for (curr_pos = 0, next_pos = 0; curr_pos < n_uniq_elements - 1; curr_pos++) {
    /* Find new split point. */
    curr_el = NUM2DBL(rb_ary_entry(uniqed_f, curr_pos));
    next_el = NUM2DBL(rb_ary_entry(sorted_f, next_pos));
    while (next_pos < n_elements && next_el <= curr_el) {
      target = rb_ary_entry(sorted_y, next_pos);
      add_sum_vec(l_sum_vec, target);
      rb_ary_push(l_target_vecs, target);
      n_l_elements++;
      sub_sum_vec(r_sum_vec, target);
      rb_ary_shift(r_target_vecs);
      n_r_elements--;
      next_el = NUM2DBL(rb_ary_entry(sorted_f, ++next_pos));
    }
    /* Calculate gain of new split. */
    l_impurity = calc_impurity_reg(criterion, l_target_vecs, l_sum_vec);
    r_impurity = calc_impurity_reg(criterion, r_target_vecs, r_sum_vec);
    gain = w_impurity - (n_l_elements * l_impurity + n_r_elements * r_impurity) / n_elements;
    /* Update optimal parameters. */
    if (gain > NUM2DBL(rb_ary_entry(opt_params, 3))) {
      rb_ary_store(opt_params, 0, DBL2NUM(l_impurity));
      rb_ary_store(opt_params, 1, DBL2NUM(r_impurity));
      rb_ary_store(opt_params, 2, DBL2NUM(0.5 * (curr_el + next_el)));
      rb_ary_store(opt_params, 3, DBL2NUM(gain));
    }
  }

  return opt_params;
}

/**
 * @!visibility private
 * Calculate impurity based on criterion.
 *
 * @overload node_impurity(criterion, y, n_classes) -> Float
 *
 * @param criterion [String] The function to calculate impurity. Supported criteria are 'gini' and 'entropy'.
 * @param y [Numo::Int32] (shape: [n_samples]) The labels.
 * @param n_classes [Integer] The number of classes.
 * @return [Float] impurity
 */
static VALUE
node_impurity_cls(VALUE self, VALUE criterion, VALUE y, VALUE n_classes)
{
  long i;
  const long n_elements = RARRAY_LEN(y);
  VALUE histogram = create_zero_vector(NUM2LONG(n_classes));

  for (i = 0; i < n_elements; i++) {
    increment_histogram(histogram, NUM2LONG(rb_ary_entry(y, i)));
  }

  return DBL2NUM(calc_impurity_cls(criterion, histogram, n_elements));
}

/**
 * @!visibility private
 * Calculate impurity based on criterion.
 *
 * @overload node_impurity(criterion, y) -> Float
 *
 * @param criterion [String] The function to calculate impurity. Supported criteria are 'mae' and 'mse'.
 * @param y [Numo::DFloat] (shape: [n_samples, n_outputs]) The taget values.
 * @return [Float] impurity
 */
static VALUE
node_impurity_reg(VALUE self, VALUE criterion, VALUE y)
{
  long i;
  const long n_elements = RARRAY_LEN(y);
  const long n_dimensions = RARRAY_LEN(rb_ary_entry(y, 0));
  VALUE sum_vec = create_zero_vector(n_dimensions);
  VALUE target_vecs = rb_ary_new();
  VALUE target;

  for (i = 0; i < n_elements; i++) {
    target = rb_ary_entry(y, i);
    add_sum_vec(sum_vec, target);
    rb_ary_push(target_vecs, target);
  }

  return DBL2NUM(calc_impurity_reg(criterion, target_vecs, sum_vec));
}

void Init_rumale(void)
{
  VALUE mRumale = rb_define_module("Rumale");
  VALUE mTree = rb_define_module_under(mRumale, "Tree");
  /**
   * Document-module: Rumale::Tree::ExtDecisionTreeClassifier
   * @!visibility private
   * The mixin module consisting of extension method for DecisionTreeClassifier class.
   * This module is used internally.
   */
  VALUE mExtDTreeCls = rb_define_module_under(mTree, "ExtDecisionTreeClassifier");
  /**
   * Document-module: Rumale::Tree::ExtDecisionTreeRegressor
   * @!visibility private
   * The mixin module consisting of extension method for DecisionTreeRegressor class.
   * This module is used internally.
   */
  VALUE mExtDTreeReg = rb_define_module_under(mTree, "ExtDecisionTreeRegressor");

  rb_define_method(mExtDTreeCls, "find_split_params", find_split_params_cls, 6);
  rb_define_method(mExtDTreeReg, "find_split_params", find_split_params_reg, 5);
  rb_define_method(mExtDTreeCls, "node_impurity", node_impurity_cls, 3);
  rb_define_method(mExtDTreeReg, "node_impurity", node_impurity_reg, 2);
}
