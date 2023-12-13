#include "tfcfm_als.h"

TFCFMWithALS::TFCFMWithALS(int missing_count)
    : FMBase(missing_count),
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
    method_name_ = append_current_time_if_test("TFCFM_ALS");
}

void TFCFMWithALS::set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda,
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

void TFCFMWithALS::set_initial_values(int seed) {
    seed *= 1000000;
    w0_ = Vector(cluster_size_, 0.0, "all");
    w_ = Matrix(cluster_size_, sum_users_items, 0.0);
    v_ = Tensor(cluster_size_, latent_dimension_, sum_users_items);
    e_ = Matrix(cluster_size_, sparse_missing_data_.nnz() - num_missing_value_, 0.0);
    q_ = Tensor(cluster_size_, latent_dimension_, sparse_missing_data_.nnz() - num_missing_value_);
    x_ = DSSTensor(sparse_missing_data_, sum_users_items);
    membership_ = Matrix(cluster_size_, rs::num_users, 1.0 / (double)cluster_size_);
    dissimilarities_ = Matrix(cluster_size_, rs::num_users, 0);

    std::mt19937_64 mt;
    for (int c = 0; c < cluster_size_; c++) {
        for (int n = 0; n < sum_users_items; n++) {
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
        for (int j = 0; j < x_.nnz(i); j++) {
            SparseVector x_element(rs::num_items, 2);
            x_element(0) = 1;
            x_element.dense_index(0) = i;
            x_element(1) = 1;
            x_element.dense_index(1) = rs::num_users + x_.dense_index(i, j);
            x_(i, j) = x_element;
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
    //     for (int j = 0; j < x_.nnz(i); j++) {
    //         for (int a = 0; a < 2; ++a) {
    //         //std::cout << "i:" << i << " j:" << j << " : " << x_(i, j) << " " << x_(i,j).dense_index(a);
    //                   //<< std::endl;
    //                   std::cout << " " << x_(i,j).dense_index(a);
    //         }
    //     }
    // }
    precompute();
    for (int i = 0; i < sparse_missing_data_rows_; i++) {
        sparse_missing_data_nnz_[i] = 0;
    }
    pointer_sparse_missing_data_cols_ = sparse_missing_data_cols_.get_values();
}

void TFCFMWithALS::precompute() {
    SparseVector tmp_x;
    Matrix tmp_v;
    for (int c = 0; c < cluster_size_; ++c) {
        for (int f = 0; f < latent_dimension_; ++f) {
            tmp_v = v_[c];
            double* tmp_q = q_[c][f].get_values();
            int l = 0;
            for (int i = 0; i < sparse_missing_data_.rows(); i++) {
                for (int j = 0; j < sparse_missing_data_.nnz(i); j++) {
                    if (sparse_missing_data_(i, j) != 0) {
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
}

void TFCFMWithALS::calculate_factors() {
    prev_w0_ = w0_;
    prev_w_ = w_;
    prev_v_ = v_;

    SparseVector tmp_x;
    double tmp_x_value;
    int tmp_x_dense_index;

    // int sparse_missing_data_rows = rs::num_users;
    // int sparse_missing_data_nnz[rs::num_users] = {};
    // std::cout << sparse_missing_data_rows<<std::endl;

    int sum_data_elements = 0;
    for (int i = 0; i < sparse_missing_data_rows; i++) {
        for (int j = 0; j < sparse_missing_data_.nnz(i); j++) {
            if (sparse_missing_data_(i, j) != 0.0) {
                sparse_missing_data_nnz[i]++;
                sum_data_elements++;
            }
        }
    }
    int sparse_missing_data_cols[sum_data_elements] = {};
    tmp_sparse_missing_data_cols = &sparse_missing_data_cols[0];
    int k = 0;
    for (int i = 0; i < sparse_missing_data_rows; i++) {
        for (int j = 0; j < sparse_missing_data_.nnz(i); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                *(tmp_sparse_missing_data_cols + k) = j;
                k++;
            }
        }
    }

    for (int c = 0; c < cluster_size_; ++c) {
        double numerator_w0 = 0;
        double denominator_w0 = 0;
        int l = 0;
        for (int i = 0; i < sparse_missing_data_rows; i++) {
            double tmp_membership = pow(membership_(c, i), fuzzifier_em_);
            for (int j = 0; j < sparse_missing_data_nnz[i]; j++) {
                numerator_w0 += tmp_membership * (e_(c, l) - w0_[c]);
                denominator_w0 += tmp_membership;
                l++;
            }
        }
        double w0a = -numerator_w0 / (denominator_w0 + reg_parameter_);
        for (l = 0; l < e_.cols(); l++) e_(c, l) += (w0a - w0_[c]);
        w0_[c] = w0a;
    }

    // 1-way interactions
    for (int c = 0; c < cluster_size_; ++c) {
        double* tmp_w = w_[c].get_values();
        double wa[sum_users_items] = {};
        double denominator_w[sum_users_items] = {};
        Vector tmp_e = e_[c];
        for (int a = 0; a < 2; ++a) {
            double numerator_w[sum_users_items] = {};
            double denominator_w[sum_users_items] = {};
            int l = 0;
            for (int i = 0; i < sparse_missing_data_rows; i++) {
                double tmp_membership = pow(membership_(c, i), fuzzifier_em_);
                for (int j = 0; j < sparse_missing_data_nnz[i]; j++) {
                    tmp_x = x_(i, tmp_sparse_missing_data_cols[l]);
                    tmp_x_value = tmp_x(a);
                    tmp_x_dense_index = tmp_x.dense_index(a);
                    numerator_w[tmp_x_dense_index] += tmp_membership * (tmp_e[l] - tmp_w[tmp_x_dense_index] * tmp_x_value) * tmp_x_value;
                    denominator_w[tmp_x_dense_index] += tmp_membership * tmp_x_value * tmp_x_value;
                    l++;
                }
            }
            for (int i = 0; i < sum_users_items; ++i) {
                if (denominator_w[i] != 0) wa[i] = -numerator_w[i] / (denominator_w[i] + reg_parameter_);
            }
            l = 0;
            for (int i = 0; i < sparse_missing_data_rows; i++) {
                double tmp_membership = pow(membership_(c, i), fuzzifier_em_);
                for (int j = 0; j < sparse_missing_data_nnz[i]; j++) {
                    tmp_x = x_(i, tmp_sparse_missing_data_cols[l]);
                    tmp_x_value = tmp_x(a);
                    tmp_x_dense_index = tmp_x.dense_index(a);
                    e_(c, l) += (wa[tmp_x_dense_index] - tmp_w[tmp_x_dense_index]) * tmp_x_value;
                    l++;
                }
            }
        }
        for (int j_ = 0; j_ < sum_users_items; ++j_) {
            w_(c, j_) = wa[j_];
        }
    }
    // 2-way interactions
    for (int c = 0; c < cluster_size_; ++c) {
        double* tmp_e = e_[c].get_values();
        for (int f = 0; f < latent_dimension_; ++f) {
            double* tmp_v = v_[c][f].get_values();
            double* tmp_q = q_[c][f].get_values();
            double va[sum_users_items] = {};
            for (int a = 0; a < 2; ++a) {
                double h_value[e_.cols()] = {};
                int l = 0;
                double numerator_v[sum_users_items] = {};
                double denominator_v[sum_users_items] = {};
                for (int i = 0; i < sparse_missing_data_rows; i++) {
                    double tmp_membership = pow(membership_(c, i), fuzzifier_em_);
                    for (int j = 0; j < sparse_missing_data_nnz[i]; j++) {
                        tmp_x = x_(i, tmp_sparse_missing_data_cols[l]);
                        tmp_x_value = tmp_x(a);
                        tmp_x_dense_index = tmp_x.dense_index(a);
                        h_value[l] = -tmp_x_value * (tmp_x_value * tmp_v[tmp_x_dense_index] - tmp_q[l]);
                        double tmp_h = h_value[l];
                        numerator_v[tmp_x_dense_index] += tmp_membership * (tmp_e[l] - tmp_v[tmp_x_dense_index] * tmp_h) * tmp_h;
                        denominator_v[tmp_x_dense_index] += tmp_membership * tmp_h * tmp_h;
                        l++;
                    }
                }
                for (int j_ = 0; j_ < sum_users_items; ++j_) {
                    if (denominator_v[j_] != 0) va[j_] = -numerator_v[j_] / (denominator_v[j_] + reg_parameter_);
                }
                l = 0;
                for (int i = 0; i < sparse_missing_data_rows; i++) {
                    for (int j = 0; j < sparse_missing_data_nnz[i]; j++) {
                        tmp_x = x_(i, tmp_sparse_missing_data_cols[l]);
                        tmp_x_value = tmp_x(a);
                        tmp_x_dense_index = tmp_x.dense_index(a);
                        tmp_e[l] += (va[tmp_x_dense_index] - tmp_v[tmp_x_dense_index]) * h_value[l];
                        tmp_q[l] += (va[tmp_x_dense_index] - tmp_v[tmp_x_dense_index]) * tmp_x_value;
                        l++;
                    }
                }
            }
            for (int j_ = 0; j_ < sum_users_items; ++j_) {
                tmp_v[j_] = va[j_];
            }
        }
    }

    for (int c = 0; c < cluster_size_; c++) {
        int l = 0;
        for (int i = 0; i < sparse_missing_data_rows; i++) {
            dissimilarities_(c, i) = 0.0;
            for (int j = 0; j < sparse_missing_data_nnz[i]; j++) {
                double tmp = 0.0;
                tmp = (sparse_missing_data_(i, tmp_sparse_missing_data_cols[l]) -
                       predict_y(x_(i, tmp_sparse_missing_data_cols[l]), w0_[c], w_[c], v_[c]));
                dissimilarities_(c, i) = tmp * tmp;
                l++;
            }
        }
    }
    calculate_membership();
}

double TFCFMWithALS::calculate_objective_value() {
    double result = 0.0;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            result += pow(membership_(c, i), fuzzifier_em_) * dissimilarities_(c, i) +
                      1 / (fuzzifier_lambda_ * (fuzzifier_em_ - 1)) * (pow(membership_(c, i), fuzzifier_em_) - 1);
        }
    }
    result += reg_parameter_ * (squared_sum(w0_) + squared_sum(w_) + squared_sum(v_)) / 2;
    return result;
}

bool TFCFMWithALS::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff =
        squared_norm(prev_w0_ - w0_) + frobenius_norm(prev_w_ - w_) + frobenius_norm(prev_v_ - v_) + frobenius_norm(prev_membership_ - membership_);
    // std::cout << " diff:" << diff << " L:" << calculate_objective_value() << "\t";
    // std::cout << "w0:" << squared_norm(prev_w0_ - w0_) << "\t";
    // std::cout << "w:" << frobenius_norm(prev_w_ - w_) << "\t";
    // std::cout << "v:" << frobenius_norm(prev_v_ - v_) << "\t";
    // std::cout << "m:" << frobenius_norm(prev_membership_ - membership_) << "\t";
    std::cout << std::endl;
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

void TFCFMWithALS::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        prediction_[index] = 0.0;
        for (int c = 0; c < cluster_size_; c++) {
            prediction_[index] += membership_(c, missing_data_indices_(index, 0)) *
                                  predict_y(x_(missing_data_indices_(index, 0), sparse_missing_data_cols_[index]), w0_[c], w_[c], v_[c]);
        }
        // std::cout << "Prediction:" << prediction_[index]
        //           << " SparseCorrectData:" << sparse_correct_data_(missing_data_indices_[index][0], missing_data_indices_[index][1]) << std::endl;
    }
}
