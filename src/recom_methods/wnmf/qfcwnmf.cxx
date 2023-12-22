#include "qfcwnmf.h"

QFCWNMF::QFCWNMF(int missing_count)
    : QFCRecom(missing_count),
      TFCRecom(missing_count),
      Recom(missing_count),
      user_factors_(),
      item_factors_(),
      prev_user_factors_(),
      prev_item_factors_(),
      transpose_sparse_missing_data_(),
      sparse_prediction_(),
      tmp_user_factors_(),
      tmp_item_factors_(),
      tmp_membership_() {
    method_name_ = append_current_time_if_test("QFCWNMF");
}

void QFCWNMF::set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_lambda) {
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
    cluster_size_ = cluster_size;
    fuzzifier_em_ = fuzzifier_em;
    fuzzifier_lambda_ = fuzzifier_lambda;
    parameters_ = {(double)latent_dimension_,(double)cluster_size_, fuzzifier_em_, fuzzifier_lambda_};
    //parameters_ = {(double)cluster_size_, fuzzifier_em_, fuzzifier_lambda_,(double)latent_dimension_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
    user_factors_ = Tensor(cluster_size_, rs::num_users, latent_dimension_);
    item_factors_ = Tensor(cluster_size_, rs::num_items, latent_dimension_);
}

void QFCWNMF::set_initial_values(int seed) {
    seed *= 1000000;
    std::mt19937_64 mt;
    for (int c = 0; c < cluster_size_; c++) {
        for (int k_i = 0; k_i < user_factors_.cols(); k_i++) {
            for (int i = 0; i < user_factors_.rows(); i++) {
                mt.seed(seed);
                // ランダムに値生成
                std::uniform_real_distribution<> rand_user_factors_(0.001, 1.0);
                user_factors_[c](i, k_i) = rand_user_factors_(mt);
                seed++;
            }
            for (int j = 0; j < item_factors_.rows(); j++) {
                mt.seed(seed);
                std::uniform_real_distribution<> rand_item_factors_(0.001, 1.0);
                item_factors_[c](j, k_i) = rand_item_factors_(mt);
                seed++;
            }
        }
    }
    membership_ = Matrix(cluster_size_, rs::num_users, 1.0 / (double)cluster_size_);
    dissimilarities_ = Matrix(cluster_size_, rs::num_users, 0);
    cluster_size_adjustments_ = Vector(cluster_size_, 1.0 / (double)cluster_size_, "all");
    // cluster_size_adjustments_ = Vector(cluster_size_, 1.0, "all");
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
    tmp_membership_ = SparseMatrix(rs::num_users, "diag");
    transpose_sparse_missing_data_ = sparse_missing_data_.transpose();
    sparse_prediction_ = sparse_missing_data_;
}

void QFCWNMF::calculate_factors() {
    prev_item_factors_ = item_factors_;
    prev_user_factors_ = user_factors_;
    // 更新式H
    for (int c = 0; c < cluster_size_; c++) {
        tmp_user_factors_ = user_factors_[c];
        tmp_item_factors_ = item_factors_[c];
        double tmp_cluster_size_adjustments = pow(cluster_size_adjustments_[c], 1 - fuzzifier_em_);
        for (int i = 0; i < rs::num_users; i++) {
            tmp_membership_(i, 0) = pow(membership_(c, i), fuzzifier_em_) * tmp_cluster_size_adjustments;
        }
        sparse_prediction_.product(tmp_user_factors_, tmp_item_factors_);
        Matrix tmp = tmp_membership_ * tmp_item_factors_;
        Matrix item_numerator = transpose_sparse_missing_data_ * tmp;
        Matrix item_denominator = sparse_prediction_.transpose() * tmp;
        for (int j = 0; j < rs::num_items; j++) {
            for (int k = 0; k < latent_dimension_; k++) {
                if (item_denominator(j, k) == 0) item_denominator(j, k) = 1.0e-07;
                item_factors_[c](j, k) *= (item_numerator(j, k) / item_denominator(j, k));
            }
        }
    }
    //  更新式W
    for (int c = 0; c < cluster_size_; c++) {
        tmp_user_factors_ = user_factors_[c];
        tmp_item_factors_ = item_factors_[c];
        sparse_prediction_.product(tmp_user_factors_, tmp_item_factors_);
        Matrix user_numerator = sparse_missing_data_ * tmp_item_factors_;
        Matrix user_denominator = sparse_prediction_ * tmp_item_factors_;
        for (int i = 0; i < rs::num_users; i++) {
            for (int k = 0; k < latent_dimension_; k++) {
                if (user_denominator(i, k) == 0) user_denominator(i, k) = 1.0e-07;
                user_factors_[c](i, k) *= (user_numerator(i, k) / user_denominator(i, k));
            }
        }
    }

    for (int c = 0; c < cluster_size_; c++) {
        user_factor_values_ = user_factors_[c].get_values();
        for (int i = 0; i < rs::num_users; i++) {
            dissimilarities_(c, i) = 0.0;
            for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                item_factor_values_ =
                    item_factors_[c].get_values() + sparse_missing_data_col_indices_[sparse_missing_data_row_pointers_[i] + j] * latent_dimension_;
                double tmp = sparse_missing_data_values_[sparse_missing_data_row_pointers_[i] + j];
                for (int k = 0; k < latent_dimension_; k++) {
                    tmp -= user_factor_values_[k] * item_factor_values_[k];
                }
                dissimilarities_(c, i) += tmp * tmp;
            }
            user_factor_values_ += latent_dimension_;
        }
    }
    calculate_membership();

    calculate_cluster_size_adjustments();
    std::cout << cluster_size_adjustments_ << std::endl;
}

double QFCWNMF::calculate_objective_value() {
    double result;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            result += pow(cluster_size_adjustments_[c], 1 - fuzzifier_em_) * pow(membership_(c, i), fuzzifier_em_) * dissimilarities_(c, i) +
                      1 / (fuzzifier_lambda_ * (fuzzifier_em_ - 1)) *
                          (pow(cluster_size_adjustments_[c], 1 - fuzzifier_em_) * pow(membership_(c, i), fuzzifier_em_) - membership_(c, i));
        }
    }
    return result;
}

bool QFCWNMF::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff = frobenius_norm(prev_user_factors_ - user_factors_) + frobenius_norm(prev_item_factors_ - item_factors_) +
                  frobenius_norm(prev_membership_ - membership_) + squared_norm(prev_cluster_size_adjustments_ - cluster_size_adjustments_);
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

void QFCWNMF::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        prediction_[index] = 0.0;
        for (int c = 0; c < cluster_size_; c++) {
            prediction_[index] += membership_(c, missing_data_indices_(index, 0)) *
                                  (user_factors_[c][missing_data_indices_(index, 0)] * item_factors_[c][missing_data_indices_(index, 1)]);
        }
    }
}
