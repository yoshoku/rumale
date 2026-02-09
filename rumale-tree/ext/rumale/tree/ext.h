#ifndef RUMALE_TREE_EXT_H
#define RUMALE_TREE_EXT_H 1

#include <math.h>
#include <string.h>
#include <float.h>

#include <ruby.h>

#include <numo/narray.h>
#include <numo/template.h>

typedef double dfloat;
typedef float sfloat;

typedef struct {
  char* criterion;
  size_t n_classes;
  double impurity;
} split_cls_opt;

#define DEF_ITER_FIND_SPLIT_CLS(tDType)                                                        \
  static tDType calc_##tDType##_entropy(                                                       \
    const tDType* histogram, const size_t n_classes, const size_t n_elements                   \
  ) {                                                                                          \
    tDType entropy = 0.0;                                                                      \
    for (size_t i = 0; i < n_classes; i++) {                                                   \
      const tDType el = histogram[i] / (tDType)n_elements;                                     \
      entropy += el * log(el + 1.0);                                                           \
    }                                                                                          \
    return -entropy;                                                                           \
  }                                                                                            \
                                                                                               \
  static tDType calc_##tDType##_gini(                                                          \
    const tDType* histogram, const size_t n_classes, const size_t n_elements                   \
  ) {                                                                                          \
    tDType gini = 0.0;                                                                         \
    for (size_t i = 0; i < n_classes; i++) {                                                   \
      const tDType el = histogram[i] / (tDType)n_elements;                                     \
      gini += el * el;                                                                         \
    }                                                                                          \
    return 1.0 - gini;                                                                         \
  }                                                                                            \
                                                                                               \
  static void iter_##tDType##_find_split_cls(na_loop_t const* lp) {                            \
    const int32_t* o = (int32_t*)NDL_PTR(lp, 0);                                               \
    const size_t n_elements = NDL_SHAPE(lp, 0)[0];                                             \
    const tDType* f = (tDType*)NDL_PTR(lp, 1);                                                 \
    const int32_t* y = (int32_t*)NDL_PTR(lp, 2);                                               \
    const split_cls_opt* opt = (split_cls_opt*)lp->opt_ptr;                                    \
    const char* criterion = opt->criterion;                                                    \
    const size_t n_classes = opt->n_classes;                                                   \
    const tDType w_impurity = (tDType)opt->impurity;                                           \
                                                                                               \
    tDType* params = (tDType*)NDL_PTR(lp, 3);                                                  \
    params[0] = 0.0;                                                                           \
    params[1] = w_impurity;                                                                    \
    params[2] = f[o[0]];                                                                       \
    params[3] = 0.0;                                                                           \
                                                                                               \
    tDType* r_histogram = ALLOCA_N(tDType, n_classes);                                         \
    tDType* l_histogram = ALLOCA_N(tDType, n_classes);                                         \
    memset(r_histogram, 0, sizeof(tDType) * n_classes);                                        \
    memset(l_histogram, 0, sizeof(tDType) * n_classes);                                        \
    for (size_t i = 0; i < n_elements; i++) {                                                  \
      r_histogram[y[o[i]]]++;                                                                  \
    }                                                                                          \
                                                                                               \
    tDType (*calc_impurity)(const tDType*, const size_t, const size_t) = calc_##tDType##_gini; \
    if (strcmp(criterion, "entropy") == 0) {                                                   \
      calc_impurity = calc_##tDType##_entropy;                                                 \
    }                                                                                          \
    size_t curr_pos = 0;                                                                       \
    size_t next_pos = 0;                                                                       \
    size_t n_l_elements = 0;                                                                   \
    size_t n_r_elements = n_elements;                                                          \
    tDType curr_el = f[o[0]];                                                                  \
    const tDType last_el = f[o[n_elements - 1]];                                               \
    while (curr_pos < n_elements && curr_el != last_el) {                                      \
      tDType next_el = f[o[next_pos]];                                                         \
      while (next_pos < n_elements && next_el == curr_el) {                                    \
        l_histogram[y[o[next_pos]]]++;                                                         \
        n_l_elements++;                                                                        \
        r_histogram[y[o[next_pos]]]--;                                                         \
        n_r_elements--;                                                                        \
        next_pos++;                                                                            \
        next_el = f[o[next_pos]];                                                              \
      }                                                                                        \
      const tDType l_impurity = calc_impurity(l_histogram, n_classes, n_l_elements);           \
      const tDType r_impurity = calc_impurity(r_histogram, n_classes, n_r_elements);           \
      const tDType gain =                                                                      \
        w_impurity - ((tDType)n_l_elements * l_impurity + (tDType)n_r_elements * r_impurity) / \
                       (tDType)n_elements;                                                     \
      if (gain > params[3]) {                                                                  \
        params[0] = l_impurity;                                                                \
        params[1] = r_impurity;                                                                \
        params[2] = 0.5 * (curr_el + next_el);                                                 \
        params[3] = gain;                                                                      \
      }                                                                                        \
      if (next_pos == n_elements) {                                                            \
        break;                                                                                 \
      }                                                                                        \
      curr_pos = next_pos;                                                                     \
      curr_el = f[o[curr_pos]];                                                                \
    }                                                                                          \
  }

typedef struct {
  char* criterion;
  size_t n_classes;
} node_impurity_cls_opt;

#define DEF_ITER_NODE_IMPURITY_CLS(tDType)                                                     \
  static void iter_##tDType##_node_impurity_cls(na_loop_t const* lp) {                         \
    const int32_t* y = (int32_t*)NDL_PTR(lp, 0);                                               \
    const size_t n_elements = NDL_SHAPE(lp, 0)[0];                                             \
    const node_impurity_cls_opt* opt = (node_impurity_cls_opt*)lp->opt_ptr;                    \
    const char* criterion = opt->criterion;                                                    \
    const size_t n_classes = opt->n_classes;                                                   \
    tDType* ret = (tDType*)NDL_PTR(lp, 1);                                                     \
    tDType* histogram = ALLOCA_N(tDType, n_classes);                                           \
    memset(histogram, 0, sizeof(tDType) * n_classes);                                          \
    for (size_t i = 0; i < n_elements; i++) {                                                  \
      histogram[y[i]]++;                                                                       \
    }                                                                                          \
    if (strcmp(criterion, "entropy") == 0) {                                                   \
      *ret = calc_##tDType##_entropy(histogram, n_classes, n_elements);                        \
    } else {                                                                                   \
      *ret = calc_##tDType##_gini(histogram, n_classes, n_elements);                           \
    }                                                                                          \
  }

typedef struct {
  char* criterion;
  double impurity;
} split_reg_opt;

#define DEF_ITER_FIND_SPLIT_REG(tDType)                                                        \
  static tDType calc_##tDType##_mae(                                                           \
    const int32_t* order, const tDType* vecs, const tDType* mean_vec, const size_t n_elements, \
    const size_t n_outputs, const size_t order_offset                                          \
  ) {                                                                                          \
    tDType sum_err = 0.0;                                                                      \
    for (size_t i = 0; i < n_elements; i++) {                                                  \
      tDType err = 0.0;                                                                        \
      for (size_t j = 0; j < n_outputs; j++) {                                                 \
        const tDType el = vecs[order[order_offset + i] * n_outputs + j] - mean_vec[j];         \
        err += fabs(el);                                                                       \
      }                                                                                        \
      err /= (tDType)n_outputs;                                                                \
      sum_err += err;                                                                          \
    }                                                                                          \
    const tDType impurity = sum_err / (tDType)n_elements;                                      \
    return impurity;                                                                           \
  }                                                                                            \
                                                                                               \
  static tDType calc_##tDType##_mse(                                                           \
    const int32_t* order, const tDType* vecs, const tDType* mean_vec, const size_t n_elements, \
    const size_t n_outputs, const size_t order_offset                                          \
  ) {                                                                                          \
    tDType sum_err = 0.0;                                                                      \
    for (size_t i = 0; i < n_elements; i++) {                                                  \
      tDType err = 0.0;                                                                        \
      for (size_t j = 0; j < n_outputs; j++) {                                                 \
        const tDType el = vecs[order[order_offset + i] * n_outputs + j] - mean_vec[j];         \
        err += el * el;                                                                        \
      }                                                                                        \
      err /= (tDType)n_outputs;                                                                \
      sum_err += err;                                                                          \
    }                                                                                          \
    const tDType impurity = sum_err / (tDType)n_elements;                                      \
    return impurity;                                                                           \
  }                                                                                            \
                                                                                               \
  static void iter_##tDType##_find_split_reg(na_loop_t const* lp) {                            \
    const int32_t* order = (int32_t*)NDL_PTR(lp, 0);                                           \
    const size_t n_elements = NDL_SHAPE(lp, 0)[0];                                             \
    const tDType* f = (tDType*)NDL_PTR(lp, 1);                                                 \
    const tDType* y = (tDType*)NDL_PTR(lp, 2);                                                 \
    const size_t n_outputs = NDL_SHAPE(lp, 2)[1];                                              \
    const split_reg_opt* opt = (split_reg_opt*)lp->opt_ptr;                                    \
    const char* criterion = opt->criterion;                                                    \
    const tDType w_impurity = (tDType)opt->impurity;                                           \
                                                                                               \
    tDType* params = (tDType*)NDL_PTR(lp, 3);                                                  \
    params[0] = 0.0;                                                                           \
    params[1] = w_impurity;                                                                    \
    params[2] = f[order[0]];                                                                   \
    params[3] = 0.0;                                                                           \
                                                                                               \
    tDType* l_sum_y = ALLOCA_N(tDType, n_outputs);                                             \
    tDType* r_sum_y = ALLOCA_N(tDType, n_outputs);                                             \
    memset(l_sum_y, 0, sizeof(tDType) * n_outputs);                                            \
    memset(r_sum_y, 0, sizeof(tDType) * n_outputs);                                            \
    for (size_t i = 0; i < n_elements; i++) {                                                  \
      for (size_t j = 0; j < n_outputs; j++) {                                                 \
        r_sum_y[j] += y[order[i] * n_outputs + j];                                             \
      }                                                                                        \
    }                                                                                          \
                                                                                               \
    tDType (*calc_impurity)(                                                                   \
      const int32_t*, const tDType*, const tDType*, const size_t, const size_t, const size_t   \
    ) = calc_##tDType##_mse;                                                                   \
    if (strcmp(criterion, "mae") == 0) {                                                       \
      calc_impurity = calc_##tDType##_mae;                                                     \
    }                                                                                          \
    size_t curr_pos = 0;                                                                       \
    size_t next_pos = 0;                                                                       \
    size_t n_l_elements = 0;                                                                   \
    size_t n_r_elements = n_elements;                                                          \
    tDType* l_mean_y = ALLOCA_N(tDType, n_outputs);                                            \
    tDType* r_mean_y = ALLOCA_N(tDType, n_outputs);                                            \
    memset(l_mean_y, 0, sizeof(tDType) * n_outputs);                                           \
    memset(r_mean_y, 0, sizeof(tDType) * n_outputs);                                           \
    tDType curr_el = f[order[0]];                                                              \
    const tDType last_el = f[order[n_elements - 1]];                                           \
    while (curr_pos < n_elements && curr_el != last_el) {                                      \
      tDType next_el = f[order[next_pos]];                                                     \
      while (next_pos < n_elements && next_el == curr_el) {                                    \
        for (size_t j = 0; j < n_outputs; j++) {                                               \
          l_sum_y[j] += y[order[next_pos] * n_outputs + j];                                    \
          r_sum_y[j] -= y[order[next_pos] * n_outputs + j];                                    \
        }                                                                                      \
        n_l_elements++;                                                                        \
        n_r_elements--;                                                                        \
        next_pos++;                                                                            \
        next_el = f[order[next_pos]];                                                          \
      }                                                                                        \
      for (size_t j = 0; j < n_outputs; j++) {                                                 \
        l_mean_y[j] = l_sum_y[j] / (tDType)n_l_elements;                                       \
        r_mean_y[j] = r_sum_y[j] / (tDType)n_r_elements;                                       \
      }                                                                                        \
      const tDType l_impurity = calc_impurity(order, y, l_mean_y, n_l_elements, n_outputs, 0); \
      const tDType r_impurity =                                                                \
        calc_impurity(order, y, r_mean_y, n_r_elements, n_outputs, next_pos);                  \
      const tDType gain =                                                                      \
        w_impurity -                                                                           \
        (n_l_elements * l_impurity + n_r_elements * r_impurity) / (tDType)n_elements;          \
                                                                                               \
      if (gain > params[3]) {                                                                  \
        params[0] = l_impurity;                                                                \
        params[1] = r_impurity;                                                                \
        params[2] = 0.5 * (curr_el + next_el);                                                 \
        params[3] = gain;                                                                      \
      }                                                                                        \
      if (next_pos == n_elements) {                                                            \
        break;                                                                                 \
      }                                                                                        \
      curr_pos = next_pos;                                                                     \
      curr_el = f[order[curr_pos]];                                                            \
    }                                                                                          \
  }

typedef struct {
  char* criterion;
} node_impurity_reg_opt;

#define DEF_ITER_NODE_IMPURITY_REG(tDType)                                                     \
  static void iter_##tDType##_node_impurity_reg(na_loop_t const* lp) {                         \
    const tDType* y = (tDType*)NDL_PTR(lp, 0);                                                 \
    const size_t n_elements = NDL_SHAPE(lp, 0)[0];                                             \
    const size_t n_outputs = NDL_SHAPE(lp, 0)[1];                                              \
    const char* criterion = ((node_impurity_reg_opt*)lp->opt_ptr)->criterion;                  \
    int32_t* order = ALLOCA_N(int32_t, n_elements);                                            \
    tDType* mean_y = ALLOCA_N(tDType, n_outputs);                                              \
    memset(mean_y, 0, sizeof(tDType) * n_outputs);                                             \
    for (size_t i = 0; i < n_elements; i++) {                                                  \
      order[i] = (int32_t)i;                                                                   \
      for (size_t j = 0; j < n_outputs; j++) {                                                 \
        mean_y[j] += y[i * n_outputs + j];                                                     \
      }                                                                                        \
    }                                                                                          \
    for (size_t j = 0; j < n_outputs; j++) {                                                   \
      mean_y[j] /= (tDType)n_elements;                                                         \
    }                                                                                          \
    tDType* ret = (tDType*)NDL_PTR(lp, 1);                                                     \
    if (strcmp(criterion, "mae") == 0) {                                                       \
      *ret = calc_##tDType##_mae(order, y, mean_y, n_elements, n_outputs, 0);                  \
    } else {                                                                                   \
      *ret = calc_##tDType##_mse(order, y, mean_y, n_elements, n_outputs, 0);                  \
    }                                                                                          \
  }

#define DEF_ITER_CHECK_SAME_VALUE(tDType, mEPS)                                                \
  static void iter_##tDType##_check_same_value(na_loop_t const* lp) {                          \
    const tDType* y = (tDType*)NDL_PTR(lp, 0);                                                 \
    const size_t n_elements = NDL_SHAPE(lp, 0)[0];                                             \
    const size_t n_outputs = NDL_SHAPE(lp, 0)[1];                                              \
    VALUE* res = (VALUE*)NDL_PTR(lp, 1);                                                       \
    *res = Qtrue;                                                                              \
    if (n_elements > 0) {                                                                      \
      for (size_t i = 1; i < n_elements; i++) {                                                \
        for (size_t j = 0; j < n_outputs; j++) {                                               \
          if (fabs(y[i * n_outputs + j] - y[j]) > mEPS) {                                      \
            *res = Qfalse;                                                                     \
            break;                                                                             \
          }                                                                                    \
        }                                                                                      \
        if (*res == Qfalse) break;                                                             \
      }                                                                                        \
    }                                                                                          \
  }

typedef struct {
  double sum_gradient;
  double sum_hessian;
  double reg_lambda;
} split_greg_opt;

#define DEF_ITER_FIND_SPLIT_GREG(tDType)                                                       \
  static void iter_##tDType##_find_split_greg(na_loop_t const* lp) {                           \
    const int32_t* o = (int32_t*)NDL_PTR(lp, 0);                                               \
    const size_t n_elements = NDL_SHAPE(lp, 0)[0];                                             \
    const tDType* f = (tDType*)NDL_PTR(lp, 1);                                                 \
    const tDType* g = (tDType*)NDL_PTR(lp, 2);                                                 \
    const tDType* h = (tDType*)NDL_PTR(lp, 3);                                                 \
    const split_greg_opt* opt = (split_greg_opt*)lp->opt_ptr;                                  \
    const tDType s_grad = (tDType)(opt->sum_gradient);                                         \
    const tDType s_hess = (tDType)(opt->sum_hessian);                                          \
    const tDType reg_lambda = (tDType)opt->reg_lambda;                                         \
    size_t curr_pos = 0;                                                                       \
    size_t next_pos = 0;                                                                       \
    tDType curr_el = f[o[0]];                                                                  \
    const tDType last_el = f[o[n_elements - 1]];                                               \
    tDType l_grad = 0.0;                                                                       \
    tDType l_hess = 0.0;                                                                       \
    tDType threshold = curr_el;                                                                \
    tDType gain_max = 0.0;                                                                     \
    while (curr_pos < n_elements && curr_el != last_el) {                                      \
      tDType next_el = f[o[next_pos]];                                                         \
      while (next_pos < n_elements && next_el == curr_el) {                                    \
        l_grad += g[o[next_pos]];                                                              \
        l_hess += h[o[next_pos]];                                                              \
        next_pos++;                                                                            \
        next_el = f[o[next_pos]];                                                              \
      }                                                                                        \
      const tDType r_grad = s_grad - l_grad;                                                   \
      const tDType r_hess = s_hess - l_hess;                                                   \
      const tDType gain = (l_grad * l_grad) / (l_hess + reg_lambda) +                          \
                          (r_grad * r_grad) / (r_hess + reg_lambda) -                          \
                          (s_grad * s_grad) / (s_hess + reg_lambda);                           \
      if (gain > gain_max) {                                                                   \
        threshold = 0.5 * (curr_el + next_el);                                                 \
        gain_max = gain;                                                                       \
      }                                                                                        \
      if (next_pos == n_elements) break;                                                       \
      curr_pos = next_pos;                                                                     \
      curr_el = f[o[curr_pos]];                                                                \
    }                                                                                          \
                                                                                               \
    tDType* params = (tDType*)NDL_PTR(lp, 4);                                                  \
    params[0] = threshold;                                                                     \
    params[1] = gain_max;                                                                      \
  }

#endif /* RUMALE_TREE_EXT_H */
