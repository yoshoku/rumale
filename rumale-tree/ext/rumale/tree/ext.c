#include "ext.h"

DEF_ITER_FIND_SPLIT_CLS(dfloat)
DEF_ITER_FIND_SPLIT_CLS(sfloat)

static VALUE find_split_params_cls(
  VALUE self, VALUE criterion, VALUE impurity, VALUE order, VALUE features, VALUE labels,
  VALUE n_classes
) {
  VALUE klass = rb_obj_class(features);
  ndfunc_arg_in_t ain[3] = { { numo_cInt32, 1 }, { klass, 1 }, { numo_cInt32, 1 } };
  size_t out_shape[1] = { 4 };
  ndfunc_arg_out_t aout[1] = { { klass, 1, out_shape } };
  na_iter_func_t iter_func = (na_iter_func_t)iter_dfloat_find_split_cls;
  if (klass == numo_cSFloat) {
    iter_func = (na_iter_func_t)iter_sfloat_find_split_cls;
  }
  ndfunc_t ndf = { iter_func, NO_LOOP, 3, 1, ain, aout };
  split_cls_opt opts = { StringValueCStr(criterion), NUM2SIZET(n_classes), NUM2DBL(impurity) };
  VALUE res = na_ndloop3(&ndf, &opts, 3, order, features, labels);
  RB_GC_GUARD(criterion);
  return res;
}

DEF_ITER_NODE_IMPURITY_CLS(dfloat)
DEF_ITER_NODE_IMPURITY_CLS(sfloat)

static VALUE
node_impurity_cls(VALUE self, VALUE criterion, VALUE y, VALUE n_classes, VALUE klass) {
  ndfunc_arg_in_t ain[1] = { { numo_cInt32, 1 } };
  ndfunc_arg_out_t aout[1] = { { klass, 0 } };
  na_iter_func_t iter_func = (na_iter_func_t)iter_dfloat_node_impurity_cls;
  if (klass == numo_cSFloat) {
    iter_func = (na_iter_func_t)iter_sfloat_node_impurity_cls;
  }
  ndfunc_t ndf = { iter_func, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };
  node_impurity_cls_opt opts = { StringValueCStr(criterion), NUM2SIZET(n_classes) };
  VALUE res = na_ndloop3(&ndf, &opts, 1, y);
  RB_GC_GUARD(criterion);
  return res;
}

static void iter_check_same_label(na_loop_t const* lp) {
  const int32_t* y = (int32_t*)NDL_PTR(lp, 0);
  const size_t n_elements = NDL_SHAPE(lp, 0)[0];
  VALUE* res = (VALUE*)NDL_PTR(lp, 1);
  *res = Qtrue;
  if (n_elements > 0) {
    int32_t label = y[0];
    for (size_t i = 0; i < n_elements; i++) {
      if (y[i] != label) {
        *res = Qfalse;
        break;
      }
    }
  }
}

static VALUE check_same_label(VALUE self, VALUE y) {
  ndfunc_arg_in_t ain[1] = { { numo_cInt32, 1 } };
  ndfunc_arg_out_t aout[1] = { { numo_cRObject, 0 } };
  ndfunc_t ndf = {
    (na_iter_func_t)iter_check_same_label, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout
  };
  return na_ndloop(&ndf, 1, y);
}

DEF_ITER_FIND_SPLIT_REG(dfloat)
DEF_ITER_FIND_SPLIT_REG(sfloat)

static VALUE find_split_params_reg(
  VALUE self, VALUE criterion, VALUE impurity, VALUE order, VALUE features, VALUE targets
) {
  VALUE klass = rb_obj_class(features);
  ndfunc_arg_in_t ain[3] = { { numo_cInt32, 1 }, { klass, 1 }, { numo_cDFloat, 2 } };
  size_t out_shape[1] = { 4 };
  ndfunc_arg_out_t aout[1] = { { klass, 1, out_shape } };
  na_iter_func_t iter_func = (na_iter_func_t)iter_dfloat_find_split_reg;
  if (klass == numo_cSFloat) {
    iter_func = (na_iter_func_t)iter_sfloat_find_split_reg;
  }
  ndfunc_t ndf = { iter_func, NO_LOOP, 3, 1, ain, aout };
  split_reg_opt opts = { StringValueCStr(criterion), NUM2DBL(impurity) };
  VALUE res = na_ndloop3(&ndf, &opts, 3, order, features, targets);
  RB_GC_GUARD(criterion);
  return res;
}

DEF_ITER_NODE_IMPURITY_REG(dfloat)
DEF_ITER_NODE_IMPURITY_REG(sfloat)

static VALUE node_impurity_reg(VALUE self, VALUE criterion, VALUE y, VALUE klass) {
  ndfunc_arg_in_t ain[1] = { { klass, 2 } };
  ndfunc_arg_out_t aout[1] = { { klass, 0 } };
  na_iter_func_t iter_func = (na_iter_func_t)iter_dfloat_node_impurity_reg;
  if (klass == numo_cSFloat) {
    iter_func = (na_iter_func_t)iter_sfloat_node_impurity_reg;
  }
  ndfunc_t ndf = { iter_func, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };
  node_impurity_reg_opt opts = { StringValueCStr(criterion) };
  VALUE res = na_ndloop3(&ndf, &opts, 1, y);
  RB_GC_GUARD(criterion);
  return res;
}

DEF_ITER_CHECK_SAME_VALUE(dfloat, DBL_EPSILON)
DEF_ITER_CHECK_SAME_VALUE(sfloat, DBL_EPSILON)

static VALUE check_same_value(VALUE self, VALUE y) {
  VALUE klass = rb_obj_class(y);
  ndfunc_arg_in_t ain[1] = { { klass, 2 } };
  ndfunc_arg_out_t aout[1] = { { numo_cRObject, 0 } };
  na_iter_func_t iter_func = (na_iter_func_t)iter_dfloat_check_same_value;
  if (klass == numo_cSFloat) {
    iter_func = (na_iter_func_t)iter_sfloat_check_same_value;
  }
  ndfunc_t ndf = { iter_func, NO_LOOP | NDF_EXTRACT, 1, 1, ain, aout };
  return na_ndloop(&ndf, 1, y);
}

DEF_ITER_FIND_SPLIT_GREG(dfloat)
DEF_ITER_FIND_SPLIT_GREG(sfloat)

static VALUE find_split_params_greg(
  VALUE self, VALUE order, VALUE features, VALUE gradients, VALUE hessians, VALUE sum_gradient,
  VALUE sum_hessian, VALUE reg_lambda
) {
  VALUE klass = rb_obj_class(features);
  ndfunc_arg_in_t ain[4] = { { numo_cInt32, 1 }, { klass, 1 }, { klass, 1 }, { klass, 1 } };
  size_t out_shape[1] = { 2 };
  ndfunc_arg_out_t aout[1] = { { klass, 1, out_shape } };
  na_iter_func_t iter_func = (na_iter_func_t)iter_dfloat_find_split_greg;
  if (klass == numo_cSFloat) {
    iter_func = (na_iter_func_t)iter_sfloat_find_split_greg;
  }
  ndfunc_t ndf = { iter_func, NO_LOOP, 4, 1, ain, aout };
  split_greg_opt opts = { NUM2DBL(sum_gradient), NUM2DBL(sum_hessian), NUM2DBL(reg_lambda) };
  VALUE params = na_ndloop3(&ndf, &opts, 4, order, features, gradients, hessians);
  return params;
}

void Init_ext(void) {
  VALUE rb_mRumale = rb_define_module("Rumale");
  VALUE rb_mTree = rb_define_module_under(rb_mRumale, "Tree");
  VALUE rb_mExtDTreeCls = rb_define_module_under(rb_mTree, "ExtDecisionTreeClassifier");
  rb_define_private_method(rb_mExtDTreeCls, "find_split_params", find_split_params_cls, 6);
  rb_define_private_method(rb_mExtDTreeCls, "node_impurity", node_impurity_cls, 4);
  rb_define_private_method(rb_mExtDTreeCls, "stop_growing?", check_same_label, 1);
  VALUE rb_mExtDTreeReg = rb_define_module_under(rb_mTree, "ExtDecisionTreeRegressor");
  rb_define_private_method(rb_mExtDTreeReg, "find_split_params", find_split_params_reg, 5);
  rb_define_private_method(rb_mExtDTreeReg, "node_impurity", node_impurity_reg, 3);
  rb_define_private_method(rb_mExtDTreeReg, "stop_growing?", check_same_value, 1);
  VALUE rb_mExtGTreeReg = rb_define_module_under(rb_mTree, "ExtGradientTreeRegressor");
  rb_define_private_method(rb_mExtGTreeReg, "find_split_params", find_split_params_greg, 7);
}
