#include "qfcfm_als.h"

QFCFMWithALS::QFCFMWithALS(int missing_count)
    : FMBase(missing_count),
      QFCRecom(missing_count),
      TFCRecom(missing_count),
      Recom(missing_count),
      w0_(),
      prev_w0_(),
      w_(),
      prev_w_(),
      v_(),
      prev_v_(),
      e_(),
      q_(),
      x_() {
    method_name_ = append_current_time_if_test("QFCFM_ALS");
}

void QFCFMWithALS::set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda,
                                  double reg_parameter) {
#if defined ARTIFICIALITY
    latent_dimension_ = latent_dimension_percentage;
#else
    if (rs::num_users > rs::num_items) {
        latent_dimension_ = std::round(rs::num_items * latent_dimension_percentage / 100);
    } else {
        latent_dimension_ = std::round(rs::num_users * latent_dimension_percentage / 100);
    }
    if (rs::steps < 50) {
        std::cerr << "MF: \"step\" should be 50 or more.";
        return;
    }
#endif
    reg_parameter_ = reg_parameter;
    cluster_size_ = cluster_size;
    fuzzifier_em_ = fuzzifier_em;
    fuzzifier_lambda_ = fuzzifier_Lambda;
    parameters_ = {(double)latent_dimension_, double(cluster_size), fuzzifier_em, fuzzifier_Lambda, reg_parameter_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
}

void QFCFMWithALS::set_initial_values(int seed) {
    seed *= 1000000;
    w0_ = Vector(cluster_size_, 0.0, "all");
    w_ = Matrix(cluster_size_, sum_users_items, 0.0);
    v_ = Tensor(cluster_size_, latent_dimension_, sum_users_items);
    e_ = Matrix(cluster_size_, missing_num_samples_, 0.0);
    q_ = Tensor(cluster_size_, latent_dimension_, missing_num_samples_);
    x_ = DSSTensor(sparse_missing_data_, sum_users_items);
    membership_ = Matrix(cluster_size_, rs::num_users, 1.0 / (double)cluster_size_);
    dissimilarities_ = Matrix(cluster_size_, rs::num_users, 0);
    cluster_size_adjustments_ = Vector(cluster_size_, 1.0 / (double)cluster_size_, "all");

    std::mt19937_64 mt;
    for (int c = 0; c < cluster_size_; c++) {
        for (int n = 0; n < rs::num_users + rs::num_items; n++) {
            for (int k = 0; k < latent_dimension_; k++) {
                mt.seed(seed);
                // ランダムに値生成
                std::uniform_real_distribution<> rand_v(-0.01, 0.01);
                v_[c](k, n) = rand_v(mt);
                // v_[c](n, k) = 1.0;
            }
        }
    }

    for (int i = 0; i < rs::num_users; i++) {
        for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
            x_(i, j) = make_one_hot_data(i, x_.dense_index(i, j));
        }
    }

    for (int k = 0; k < rs::num_users; k++) {
        double tmp_Mem[cluster_size_];
        tmp_Mem[cluster_size_ - 1] = 1.0;
        for (int i = 0; i < cluster_size_ - 1; i++) {
            mt.seed(seed);
            std::uniform_real_distribution<> rand_p(0.01, 1.0 / (double)cluster_size_);
            tmp_Mem[i] = rand_p(mt);
            tmp_Mem[cluster_size_ - 1] -= tmp_Mem[i];
            seed++;
        }
        // [0, 99] 範囲の一様乱数
        for (int i = 0; i < cluster_size_; i++) {
            mt.seed(seed);
            std::uniform_int_distribution<> rand100(0, 99);
            int r = rand100(mt) % (1 + i);
            double tmp = tmp_Mem[i];
            tmp_Mem[i] = tmp_Mem[r];
            tmp_Mem[r] = tmp;
            seed++;
        }
        for (int i = 0; i < cluster_size_; i++) {
            membership_(i, k) = tmp_Mem[i];
        }
    }

    // データ表示
    // for (int i = 0; i < rs::num_users; i++) {
    //     for (int j = 0; j < x_(i, "row"); j++) {
    //         std::cout << "i:" << i << " j:" << j << " : " << x_(i, j)
    //                   << std::endl;
    //     }
    // }
    precompute();
}

void QFCFMWithALS::precompute() {
    SparseVector tmp_x;
    Matrix tmp_v;
    for (int c = 0; c < cluster_size_; ++c) {
        for (int f = 0; f < latent_dimension_; ++f) {
            tmp_v = v_[c];
            double* tmp_q = q_[c][f].get_values();
            int l = 0;
            for (int i = 0; i < rs::num_users; i++) {
                for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                    tmp_x = x_(i, j);
                    if (f == 0) e_(c, l) = predict_y(tmp_x, w0_[c], w_[c], tmp_v) - sparse_missing_data_(i, j);
                    tmp_q[l] = 0.0;
                    for (int a = 0; a < 2; ++a) {
                        tmp_q[l] += tmp_x(a) * tmp_v(f, tmp_x.dense_index(a));
                    }
                    l++;
                }
            }
        }
    }
}

void QFCFMWithALS::calculate_factors() {
    prev_v_ = v_;
    prev_w_ = w_;
    prev_w0_ = w0_;

    SparseVector tmp_x;
    double tmp_x_value;
    int tmp_x_dense_index;

    for (int c = 0; c < cluster_size_; ++c) {
        double tmp_cluster_size_adjustments = pow(cluster_size_adjustments_[c], 1 - fuzzifier_em_);
        double numerator_w0 = 0;
        double denominator_w0 = 0;
        int l = 0;
        for (int i = 0; i < rs::num_users; i++) {
            double tmp_membership_times_cluster_size_adjustments = tmp_cluster_size_adjustments * pow(membership_(c, i), fuzzifier_em_);
            for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                numerator_w0 += tmp_membership_times_cluster_size_adjustments * (e_(c, l) - w0_[c]);
                denominator_w0 += tmp_membership_times_cluster_size_adjustments;
                l++;
            }
        }
        double w0a = -numerator_w0 / (denominator_w0 + reg_parameter_/2);
        for (l = 0; l < missing_num_samples_; l++) e_(c, l) += (w0a - w0_[c]);
        w0_[c] = w0a;
    }

    // 1-way interactions
    for (int c = 0; c < cluster_size_; ++c) {
        double* tmp_w = w_[c].get_values();
        double tmp_cluster_size_adjustments = pow(cluster_size_adjustments_[c], 1 - fuzzifier_em_);
        double wa[sum_users_items] = {};
        double denominator_w[sum_users_items] = {};
        Vector tmp_e = e_[c];
        for (int a = 0; a < 2; ++a) {
            double numerator_w[sum_users_items] = {};
            double denominator_w[sum_users_items] = {};
            int l = 0;
            for (int i = 0; i < rs::num_users; i++) {
                double tmp_membership_times_cluster_size_adjustments = tmp_cluster_size_adjustments * pow(membership_(c, i), fuzzifier_em_);
                // double tmp_membership_times_cluster_size_adjustments = pow(membership_(c, i), fuzzifier_em_);
                for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                    tmp_x = x_(i, j);
                    tmp_x_value = tmp_x(a);
                    tmp_x_dense_index = tmp_x.dense_index(a);
                    numerator_w[tmp_x_dense_index] +=
                        tmp_membership_times_cluster_size_adjustments * (tmp_e[l] - tmp_w[tmp_x_dense_index] * tmp_x_value) * tmp_x_value;
                    denominator_w[tmp_x_dense_index] += tmp_membership_times_cluster_size_adjustments * tmp_x_value * tmp_x_value;
                    l++;
                }
            }
            for (int i = 0; i < sum_users_items; ++i) {
                if (denominator_w[i] != 0) wa[i] = -numerator_w[i] / (denominator_w[i] + reg_parameter_/2);
            }
            l = 0;
            for (int i = 0; i < rs::num_users; i++) {
                for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                    tmp_x = x_(i, j);
                    tmp_x_value = tmp_x(a);
                    tmp_x_dense_index = tmp_x.dense_index(a);
                    e_(c, l) += (wa[tmp_x_dense_index] - tmp_w[tmp_x_dense_index]) * tmp_x_value;
                    l++;
                }
            }
        }
        for (int a = 0; a < sum_users_items; ++a) {
            w_(c, a) = wa[a];
        }
    }

    // 2-way interactions
    for (int c = 0; c < cluster_size_; ++c) {
        double* tmp_e = e_[c].get_values();
        double tmp_cluster_size_adjustments = pow(cluster_size_adjustments_[c], 1 - fuzzifier_em_);
        for (int f = 0; f < latent_dimension_; ++f) {
            double* tmp_v = v_[c][f].get_values();
            double* tmp_q = q_[c][f].get_values();
            double va[sum_users_items] = {};
            for (int a = 0; a < 2; ++a) {
                double h_value[missing_num_samples_] = {};
                int l = 0;
                double numerator_v[sum_users_items] = {};
                double denominator_v[sum_users_items] = {};
                for (int i = 0; i < rs::num_users; i++) {
                    double tmp_membership_times_cluster_size_adjustments = tmp_cluster_size_adjustments * pow(membership_(c, i), fuzzifier_em_);
                    for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                        tmp_x = x_(i, j);
                        tmp_x_value = tmp_x(a);
                        tmp_x_dense_index = tmp_x.dense_index(a);
                        h_value[l] = -tmp_x_value * (tmp_x_value * tmp_v[tmp_x_dense_index] - tmp_q[l]);
                        double tmp_h = h_value[l];
                        numerator_v[tmp_x_dense_index] +=
                            tmp_membership_times_cluster_size_adjustments * (tmp_e[l] - tmp_v[tmp_x_dense_index] * tmp_h) * tmp_h;
                        denominator_v[tmp_x_dense_index] += tmp_membership_times_cluster_size_adjustments * tmp_h * tmp_h;
                        l++;
                    }
                }
                for (int i = 0; i < sum_users_items; ++i) {
                    if (denominator_v[i] != 0) va[i] = -numerator_v[i] / (denominator_v[i] + reg_parameter_/2);
                }
                l = 0;
                for (int i = 0; i < rs::num_users; i++) {
                    for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                        tmp_x = x_(i, j);
                        tmp_x_value = tmp_x(a);
                        tmp_x_dense_index = tmp_x.dense_index(a);
                        double difference_v = va[tmp_x_dense_index] - tmp_v[tmp_x_dense_index];
                        tmp_e[l] += difference_v * h_value[l];
                        tmp_q[l] += difference_v * tmp_x_value;
                        l++;
                    }
                }
            }
            for (int i = 0; i < sum_users_items; ++i) {
                tmp_v[i] = va[i];
            }
        }
    }

    for (int c = 0; c < cluster_size_; c++) {
        int l = 0;
        for (int i = 0; i < rs::num_users; i++) {
            dissimilarities_(c, i) = 0.0;
            for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                double tmp = 0.0;
                tmp = (sparse_missing_data_(i, j) - predict_y(x_(i, j), w0_[c], w_[c], v_[c]));
                dissimilarities_(c, i) = tmp * tmp;
                l++;
            }
        }
    }
    calculate_membership();
    calculate_cluster_size_adjustments();
}

double QFCFMWithALS::calculate_objective_value() {
    double result = 0;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            result += pow(cluster_size_adjustments_[c], 1 - fuzzifier_em_) * pow(membership_(c, i), fuzzifier_em_) * dissimilarities_(c, i) +
                      1 / (fuzzifier_lambda_ * (fuzzifier_em_ - 1)) *
                          (pow(cluster_size_adjustments_[c], 1 - fuzzifier_em_) * pow(membership_(c, i), fuzzifier_em_) - membership_(c, i));
        }
    }
    result += reg_parameter_ * (squared_sum(w0_) + squared_sum(w_) + squared_sum(v_))/2;
    return result;
}

bool QFCFMWithALS::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff = squared_norm(prev_w0_ - w0_) + frobenius_norm(prev_w_ - w_) + frobenius_norm(prev_v_ - v_) +
                  frobenius_norm(prev_membership_ - membership_) + squared_norm(prev_cluster_size_adjustments_ - cluster_size_adjustments_);
    // std::cout << " diff:" << diff << " L:" << calculate_objective_value() << "\t";
    // std::cout << "w0:" << squared_norm(prev_w0_ - w0_) << "\t";
    // std::cout << "w:" << frobenius_norm(prev_w_ - w_) << "\t";
    // std::cout << "v:" << frobenius_norm(prev_v_ - v_) << "\t";
    // std::cout << "m:" << frobenius_norm(prev_membership_ - membership_) << "\t";
#else
    objective_value_ = calculate_objective_value();
    double diff = (prev_objective_value_ - objective_value_) / prev_objective_value_;
    prev_objective_value_ = objective_value_;
#endif
    if (std::isfinite(diff)) {
        if (diff < rs::convergence_criteria) {
            result = true;
        }
    } else {
        error_detected_ = true;
    }
    return result;
}

void QFCFMWithALS::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        prediction_[index] = 0.0;
        SparseVector tmp = make_one_hot_data(missing_data_indices_(index, 0), sparse_missing_data_cols_[index]);
        for (int c = 0; c < cluster_size_; c++) {
            prediction_[index] += membership_(c, missing_data_indices_(index, 0)) * predict_y(tmp, w0_[c], w_[c], v_[c]);
        }
        // std::cout << "Prediction:" << prediction_[index]
        //           << " SparseCorrectData:" << sparse_correct_data_(missing_data_indices_[index][0], missing_data_indices_[index][1]) << std::endl;
    }
}
