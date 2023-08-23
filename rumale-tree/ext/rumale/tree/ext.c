#include "ext.h"

double* alloc_dbl_array(const long n_dimensions) {
  double* arr = ALLOC_N(double, n_dimensions);
  memset(arr, 0, n_dimensions * sizeof(double));
  return arr;
}

double calc_gini_coef(double* histogram, const long n_elements, const long n_classes) {
  double gini = 0.0;

  for (long i = 0; i < n_classes; i++) {
    double el = histogram[i] / n_elements;
    gini += el * el;
  }

  return 1.0 - gini;
}

double calc_entropy(double* histogram, const long n_elements, const long n_classes) {
  double entropy = 0.0;

  for (long i = 0; i < n_classes; i++) {
    double el = histogram[i] / n_elements;
    entropy += el * log(el + 1.0);
  }

  return -entropy;
}

VALUE
calc_mean_vec(double* sum_vec, const long n_dimensions, const long n_elements) {
  VALUE mean_vec = rb_ary_new2(n_dimensions);

  for (long i = 0; i < n_dimensions; i++) {
    rb_ary_store(mean_vec, i, DBL2NUM(sum_vec[i] / n_elements));
  }

  return mean_vec;
}

double calc_vec_mae(VALUE vec_a, VALUE vec_b) {
  const long n_dimensions = RARRAY_LEN(vec_a);
  double sum = 0.0;

  for (long i = 0; i < n_dimensions; i++) {
    double diff = NUM2DBL(rb_ary_entry(vec_a, i)) - NUM2DBL(rb_ary_entry(vec_b, i));
    sum += fabs(diff);
  }

  return sum / n_dimensions;
}

double calc_vec_mse(VALUE vec_a, VALUE vec_b) {
  const long n_dimensions = RARRAY_LEN(vec_a);
  double sum = 0.0;

  for (long i = 0; i < n_dimensions; i++) {
    double diff = NUM2DBL(rb_ary_entry(vec_a, i)) - NUM2DBL(rb_ary_entry(vec_b, i));
    sum += diff * diff;
  }

  return sum / n_dimensions;
}

double calc_mae(VALUE target_vecs, VALUE mean_vec) {
  const long n_elements = RARRAY_LEN(target_vecs);
  double sum = 0.0;

  for (long i = 0; i < n_elements; i++) {
    sum += calc_vec_mae(rb_ary_entry(target_vecs, i), mean_vec);
  }

  return sum / n_elements;
}

double calc_mse(VALUE target_vecs, VALUE mean_vec) {
  const long n_elements = RARRAY_LEN(target_vecs);
  double sum = 0.0;

  for (long i = 0; i < n_elements; i++) {
    sum += calc_vec_mse(rb_ary_entry(target_vecs, i), mean_vec);
  }

  return sum / n_elements;
}

double calc_impurity_cls(const char* criterion, double* histogram, const long n_elements, const long n_classes) {
  if (strcmp(criterion, "entropy") == 0) {
    return calc_entropy(histogram, n_elements, n_classes);
  }
  return calc_gini_coef(histogram, n_elements, n_classes);
}

double calc_impurity_reg(const char* criterion, VALUE target_vecs, double* sum_vec) {
  const long n_elements = RARRAY_LEN(target_vecs);
  const long n_dimensions = RARRAY_LEN(rb_ary_entry(target_vecs, 0));
  VALUE mean_vec = calc_mean_vec(sum_vec, n_dimensions, n_elements);

  if (strcmp(criterion, "mae") == 0) {
    return calc_mae(target_vecs, mean_vec);
  }
  return calc_mse(target_vecs, mean_vec);
}

void add_sum_vec(double* sum_vec, VALUE target) {
  const long n_dimensions = RARRAY_LEN(target);

  for (long i = 0; i < n_dimensions; i++) {
    sum_vec[i] += NUM2DBL(rb_ary_entry(target, i));
  }
}

void sub_sum_vec(double* sum_vec, VALUE target) {
  const long n_dimensions = RARRAY_LEN(target);

  for (long i = 0; i < n_dimensions; i++) {
    sum_vec[i] -= NUM2DBL(rb_ary_entry(target, i));
  }
}

/**
 * @!visibility private
 */
typedef struct {
  char* criterion;
  long n_classes;
  double impurity;
} split_opts_cls;

/**
 * @!visibility private
 */
static void iter_find_split_params_cls(na_loop_t const* lp) {
  const int32_t* o = (int32_t*)NDL_PTR(lp, 0);
  const double* f = (double*)NDL_PTR(lp, 1);
  const int32_t* y = (int32_t*)NDL_PTR(lp, 2);
  const long n_elements = NDL_SHAPE(lp, 0)[0];
  const char* criterion = ((split_opts_cls*)lp->opt_ptr)->criterion;
  const long n_classes = ((split_opts_cls*)lp->opt_ptr)->n_classes;
  const double w_impurity = ((split_opts_cls*)lp->opt_ptr)->impurity;
  double* params = (double*)NDL_PTR(lp, 3);
  long curr_pos = 0;
  long next_pos = 0;
  long n_l_elements = 0;
  long n_r_elements = n_elements;
  double curr_el = f[o[0]];
  double last_el = f[o[n_elements - 1]];
  double next_el;
  double l_impurity;
  double r_impurity;
  double gain;
  double* l_histogram = alloc_dbl_array(n_classes);
  double* r_histogram = alloc_dbl_array(n_classes);

  /* Initialize optimal parameters. */
  params[0] = 0.0;        /* left impurity */
  params[1] = w_impurity; /* right impurity */
  params[2] = curr_el;    /* threshold */
  params[3] = 0.0;        /* gain */

  /* Initialize child node variables. */
  for (long i = 0; i < n_elements; i++) {
    r_histogram[y[o[i]]] += 1.0;
  }

  /* Find optimal parameters. */
  while (curr_pos < n_elements && curr_el != last_el) {
    next_el = f[o[next_pos]];
    while (next_pos < n_elements && next_el == curr_el) {
      l_histogram[y[o[next_pos]]] += 1;
      n_l_elements++;
      r_histogram[y[o[next_pos]]] -= 1;
      n_r_elements--;
      next_pos++;
      next_el = f[o[next_pos]];
    }
    /* Calculate gain of new split. */
    l_impurity = calc_impurity_cls(criterion, l_histogram, n_l_elements, n_classes);
    r_impurity = calc_impurity_cls(criterion, r_histogram, n_r_elements, n_classes);
    gain = w_impurity - (n_l_elements * l_impurity + n_r_elements * r_impurity) / n_elements;
    /* Update optimal parameters. */
    if (gain > params[3]) {
      params[0] = l_impurity;
      params[1] = r_impurity;
      params[2] = 0.5 * (curr_el + next_el);
      params[3] = gain;
    }
    if (next_pos == n_elements) break;
    curr_pos = next_pos;
    curr_el = f[o[curr_pos]];
  }

  xfree(l_histogram);
  xfree(r_histogram);
}

/**
 * @!visibility private
 * Find for split point with maximum information gain.
 *
 * @overload find_split_params(criterion, impurity, order, features, labels, n_classes) -> Array<Float>
 *
 * @param criterion [String] The function to evaluate spliting point. Supported criteria are 'gini' and 'entropy'.
 * @param impurity [Float] The impurity of whole dataset.
 * @param order [Numo::Int32] (shape: [n_elements]) The element indices sorted according to feature values.
 * @param features [Numo::DFloat] (shape: [n_elements]) The feature values.
 * @param labels [Numo::Int32] (shape: [n_elements]) The labels.
 * @param n_classes [Integer] The number of classes.
 * @return [Array<Float>] The array consists of optimal parameters including impurities of child nodes, threshold, and gain.
 */
static VALUE find_split_params_cls(VALUE self, VALUE criterion, VALUE impurity, VALUE order, VALUE features, VALUE labels,
                                   VALUE n_classes) {
  ndfunc_arg_in_t ain[3] = {{numo_cInt32, 1}, {numo_cDFloat, 1}, {numo_cInt32, 1}};
  size_t out_shape[1] = {4};
  ndfunc_arg_out_t aout[1] = {{numo_cDFloat, 1, out_shape}};
  ndfunc_t ndf = {(na_iter_func_t)iter_find_split_params_cls, NO_LOOP, 3, 1, ain, aout};
  split_opts_cls opts = {StringValuePtr(criterion), NUM2LONG(n_classes), NUM2DBL(impurity)};
  VALUE params = na_ndloop3(&ndf, &opts, 3, order, features, labels);
  RB_GC_GUARD(criterion);
  return params;
}

/**
 * @!visibility private
 */
typedef struct {
  char* criterion;
  double impurity;
} split_opts_reg;

/**
 * @!visibility private
 */
static void iter_find_split_params_reg(na_loop_t const* lp) {
  const int32_t* o = (int32_t*)NDL_PTR(lp, 0);
  const double* f = (double*)NDL_PTR(lp, 1);
  const double* y = (double*)NDL_PTR(lp, 2);
  const long n_elements = NDL_SHAPE(lp, 0)[0];
  const long n_outputs = NDL_SHAPE(lp, 2)[1];
  const char* criterion = ((split_opts_reg*)lp->opt_ptr)->criterion;
  const double w_impurity = ((split_opts_reg*)lp->opt_ptr)->impurity;
  double* params = (double*)NDL_PTR(lp, 3);
  long curr_pos = 0;
  long next_pos = 0;
  long n_l_elements = 0;
  long n_r_elements = n_elements;
  double curr_el = f[o[0]];
  double last_el = f[o[n_elements - 1]];
  double next_el;
  double l_impurity;
  double r_impurity;
  double gain;
  double* l_sum_vec = alloc_dbl_array(n_outputs);
  double* r_sum_vec = alloc_dbl_array(n_outputs);
  double target_var;
  VALUE l_target_vecs = rb_ary_new();
  VALUE r_target_vecs = rb_ary_new();
  VALUE target;

  /* Initialize optimal parameters. */
  params[0] = 0.0;        /* left impurity */
  params[1] = w_impurity; /* right impurity */
  params[2] = curr_el;    /* threshold */
  params[3] = 0.0;        /* gain */

  /* Initialize child node variables. */
  for (long i = 0; i < n_elements; i++) {
    target = rb_ary_new2(n_outputs);
    for (long j = 0; j < n_outputs; j++) {
      target_var = y[o[i] * n_outputs + j];
      rb_ary_store(target, j, DBL2NUM(target_var));
      r_sum_vec[j] += target_var;
    }
    rb_ary_push(r_target_vecs, target);
  }

  /* Find optimal parameters. */
  while (curr_pos < n_elements && curr_el != last_el) {
    next_el = f[o[next_pos]];
    while (next_pos < n_elements && next_el == curr_el) {
      target = rb_ary_shift(r_target_vecs);
      n_r_elements--;
      sub_sum_vec(r_sum_vec, target);
      rb_ary_push(l_target_vecs, target);
      n_l_elements++;
      add_sum_vec(l_sum_vec, target);
      next_pos++;
      next_el = f[o[next_pos]];
    }
    /* Calculate gain of new split. */
    l_impurity = calc_impurity_reg(criterion, l_target_vecs, l_sum_vec);
    r_impurity = calc_impurity_reg(criterion, r_target_vecs, r_sum_vec);
    gain = w_impurity - (n_l_elements * l_impurity + n_r_elements * r_impurity) / n_elements;
    /* Update optimal parameters. */
    if (gain > params[3]) {
      params[0] = l_impurity;
      params[1] = r_impurity;
      params[2] = 0.5 * (curr_el + next_el);
      params[3] = gain;
    }
    if (next_pos == n_elements) break;
    curr_pos = next_pos;
    curr_el = f[o[curr_pos]];
  }

  xfree(l_sum_vec);
  xfree(r_sum_vec);
}

/**
 * @!visibility private
 * Find for split point with maximum information gain.
 *
 * @overload find_split_params(criterion, impurity, order, features, targets) -> Array<Float>
 *
 * @param criterion [String] The function to evaluate spliting point. Supported criteria are 'mae' and 'mse'.
 * @param impurity [Float] The impurity of whole dataset.
 * @param order [Numo::Int32] (shape: [n_samples]) The element indices sorted according to feature values in ascending order.
 * @param features [Numo::DFloat] (shape: [n_samples]) The feature values.
 * @param targets [Numo::DFloat] (shape: [n_samples, n_outputs]) The target values.
 * @return [Array<Float>] The array consists of optimal parameters including impurities of child nodes, threshold, and gain.
 */
static VALUE find_split_params_reg(VALUE self, VALUE criterion, VALUE impurity, VALUE order, VALUE features, VALUE targets) {
  ndfunc_arg_in_t ain[3] = {{numo_cInt32, 1}, {numo_cDFloat, 1}, {numo_cDFloat, 2}};
  size_t out_shape[1] = {4};
  ndfunc_arg_out_t aout[1] = {{numo_cDFloat, 1, out_shape}};
  ndfunc_t ndf = {(na_iter_func_t)iter_find_split_params_reg, NO_LOOP, 3, 1, ain, aout};
  split_opts_reg opts = {StringValuePtr(criterion), NUM2DBL(impurity)};
  VALUE params = na_ndloop3(&ndf, &opts, 3, order, features, targets);
  RB_GC_GUARD(criterion);
  return params;
}

/**
 * @!visibility private
 */
static void iter_find_split_params_grad_reg(na_loop_t const* lp) {
  const int32_t* o = (int32_t*)NDL_PTR(lp, 0);
  const double* f = (double*)NDL_PTR(lp, 1);
  const double* g = (double*)NDL_PTR(lp, 2);
  const double* h = (double*)NDL_PTR(lp, 3);
  const double s_grad = ((double*)lp->opt_ptr)[0];
  const double s_hess = ((double*)lp->opt_ptr)[1];
  const double reg_lambda = ((double*)lp->opt_ptr)[2];
  const long n_elements = NDL_SHAPE(lp, 0)[0];
  double* params = (double*)NDL_PTR(lp, 4);
  long curr_pos = 0;
  long next_pos = 0;
  double curr_el = f[o[0]];
  double last_el = f[o[n_elements - 1]];
  double next_el;
  double l_grad = 0.0;
  double l_hess = 0.0;
  double r_grad;
  double r_hess;
  double threshold = curr_el;
  double gain_max = 0.0;
  double gain;

  /* Find optimal parameters. */
  while (curr_pos < n_elements && curr_el != last_el) {
    next_el = f[o[next_pos]];
    while (next_pos < n_elements && next_el == curr_el) {
      l_grad += g[o[next_pos]];
      l_hess += h[o[next_pos]];
      next_pos++;
      next_el = f[o[next_pos]];
    }
    /* Calculate gain of new split. */
    r_grad = s_grad - l_grad;
    r_hess = s_hess - l_hess;
    gain = (l_grad * l_grad) / (l_hess + reg_lambda) + (r_grad * r_grad) / (r_hess + reg_lambda) -
           (s_grad * s_grad) / (s_hess + reg_lambda);
    /* Update optimal parameters. */
    if (gain > gain_max) {
      threshold = 0.5 * (curr_el + next_el);
      gain_max = gain;
    }
    if (next_pos == n_elements) {
      break;
    }
    curr_pos = next_pos;
    curr_el = f[o[curr_pos]];
  }

  params[0] = threshold;
  params[1] = gain_max;
}

/**
 * @!visibility private
 * Find for split point with maximum information gain.
 *
 * @overload find_split_params(order, features, gradients, hessians, sum_gradient, sum_hessian, reg_lambda) -> Array<Float>
 *   @param order [Numo::Int32] (shape: [n_elements]) The element indices sorted according to feature values.
 *   @param features [Numo::DFloat] (shape: [n_elements]) The feature values.
 *   @param gradients [Numo::DFloat] (shape: [n_elements]) The gradient values.
 *   @param hessians [Numo::DFloat] (shape: [n_elements]) The hessian values.
 *   @param sum_gradient [Float] The sum of gradient values.
 *   @param sum_hessian [Float] The sum of hessian values.
 *   @param reg_lambda [Float] The L2 regularization term on weight.
 * @return [Array<Float>] The array consists of optimal parameters including threshold and gain.
 */
static VALUE find_split_params_grad_reg(VALUE self, VALUE order, VALUE features, VALUE gradients, VALUE hessians,
                                        VALUE sum_gradient, VALUE sum_hessian, VALUE reg_lambda) {
  ndfunc_arg_in_t ain[4] = {{numo_cInt32, 1}, {numo_cDFloat, 1}, {numo_cDFloat, 1}, {numo_cDFloat, 1}};
  size_t out_shape[1] = {2};
  ndfunc_arg_out_t aout[1] = {{numo_cDFloat, 1, out_shape}};
  ndfunc_t ndf = {(na_iter_func_t)iter_find_split_params_grad_reg, NO_LOOP, 4, 1, ain, aout};
  double opts[3] = {NUM2DBL(sum_gradient), NUM2DBL(sum_hessian), NUM2DBL(reg_lambda)};
  VALUE params = na_ndloop3(&ndf, opts, 4, order, features, gradients, hessians);
  return params;
}

/**
 * @!visibility private
 */
typedef struct {
  char* criterion;
  long n_classes;
} node_impurity_cls_opts;

/**
 * @!visibility private
 */
static void iter_node_impurity_cls(na_loop_t const* lp) {
  const int32_t* y = (int32_t*)NDL_PTR(lp, 0);
  const char* criterion = ((node_impurity_cls_opts*)lp->opt_ptr)->criterion;
  const long n_classes = ((node_impurity_cls_opts*)lp->opt_ptr)->n_classes;
  const long n_elements = NDL_SHAPE(lp, 0)[0];
  double* ret = (double*)NDL_PTR(lp, 1);
  double* histogram = alloc_dbl_array(n_classes);
  for (long i = 0; i < n_elements; i++) histogram[y[i]] += 1;
  *ret = calc_impurity_cls(criterion, histogram, n_elements, n_classes);
  xfree(histogram);
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
static VALUE node_impurity_cls(VALUE self, VALUE criterion, VALUE y, VALUE n_classes) {
  ndfunc_arg_in_t ain[1] = {{numo_cInt32, 1}};
  ndfunc_arg_out_t aout[1] = {{numo_cDFloat, 0}};
  ndfunc_t ndf = {(na_iter_func_t)iter_node_impurity_cls, NDF_EXTRACT, 1, 1, ain, aout};
  node_impurity_cls_opts opts = {StringValuePtr(criterion), NUM2LONG(n_classes)};
  VALUE ret = na_ndloop3(&ndf, &opts, 1, y);
  RB_GC_GUARD(criterion);
  return ret;
}

/**
 * @!visibility private
 */
static void iter_check_same_label(na_loop_t const* lp) {
  const int32_t* y = (int32_t*)NDL_PTR(lp, 0);
  const long n_elements = NDL_SHAPE(lp, 0)[0];
  int32_t* ret = (int32_t*)NDL_PTR(lp, 1);
  *ret = 1;
  if (n_elements > 0) {
    int32_t label = y[0];
    for (long i = 0; i < n_elements; i++) {
      if (y[i] != label) {
        *ret = 0;
        break;
      }
    }
  }
}

/**
 * @!visibility private
 * Check all elements have the save value.
 *
 * @overload check_same_label(y) -> Boolean
 *
 * @param y [Numo::Int32] (shape: [n_samples]) The labels.
 * @return [Boolean]
 */
static VALUE check_same_label(VALUE self, VALUE y) {
  ndfunc_arg_in_t ain[1] = {{numo_cInt32, 1}};
  ndfunc_arg_out_t aout[1] = {{numo_cInt32, 0}};
  ndfunc_t ndf = {(na_iter_func_t)iter_check_same_label, NDF_EXTRACT, 1, 1, ain, aout};
  VALUE ret = na_ndloop(&ndf, 1, y);
  return (NUM2INT(ret) == 1 ? Qtrue : Qfalse);
}

/**
 * @!visibility private
 * Calculate impurity based on criterion.
 *
 * @overload node_impurity(criterion, y) -> Float
 *
 * @param criterion [String] The function to calculate impurity. Supported criteria are 'mae' and 'mse'.
 * @param y [Array<Float>] (shape: [n_samples, n_outputs]) The taget values.
 * @return [Float] impurity
 */
static VALUE node_impurity_reg(VALUE self, VALUE criterion, VALUE y) {
  const long n_elements = RARRAY_LEN(y);
  const long n_outputs = RARRAY_LEN(rb_ary_entry(y, 0));
  double* sum_vec = alloc_dbl_array(n_outputs);
  VALUE target_vecs = rb_ary_new();

  for (long i = 0; i < n_elements; i++) {
    VALUE target = rb_ary_entry(y, i);
    add_sum_vec(sum_vec, target);
    rb_ary_push(target_vecs, target);
  }

  VALUE ret = DBL2NUM(calc_impurity_reg(StringValuePtr(criterion), target_vecs, sum_vec));
  xfree(sum_vec);
  RB_GC_GUARD(criterion);
  return ret;
}

void Init_ext(void) {
  VALUE rb_mRumale = rb_define_module("Rumale");
  VALUE rb_mTree = rb_define_module_under(rb_mRumale, "Tree");

  /**
   * Document-module: Rumale::Tree::ExtDecisionTreeClassifier
   * @!visibility private
   * The mixin module consisting of extension method for DecisionTreeClassifier class.
   * This module is used internally.
   */
  VALUE rb_mExtDTreeCls = rb_define_module_under(rb_mTree, "ExtDecisionTreeClassifier");
  /**
   * Document-module: Rumale::Tree::ExtDecisionTreeRegressor
   * @!visibility private
   * The mixin module consisting of extension method for DecisionTreeRegressor class.
   * This module is used internally.
   */
  VALUE rb_mExtDTreeReg = rb_define_module_under(rb_mTree, "ExtDecisionTreeRegressor");
  /**
   * Document-module: Rumale::Tree::ExtGradientTreeRegressor
   * @!visibility private
   * The mixin module consisting of extension method for GradientTreeRegressor class.
   * This module is used internally.
   */
  VALUE rb_mExtGTreeReg = rb_define_module_under(rb_mTree, "ExtGradientTreeRegressor");

  rb_define_private_method(rb_mExtDTreeCls, "find_split_params", find_split_params_cls, 6);
  rb_define_private_method(rb_mExtDTreeReg, "find_split_params", find_split_params_reg, 5);
  rb_define_private_method(rb_mExtGTreeReg, "find_split_params", find_split_params_grad_reg, 7);
  rb_define_private_method(rb_mExtDTreeCls, "node_impurity", node_impurity_cls, 3);
  rb_define_private_method(rb_mExtDTreeCls, "stop_growing?", check_same_label, 1);
  rb_define_private_method(rb_mExtDTreeReg, "node_impurity", node_impurity_reg, 2);
}
