#include "tfcwnmf.h"

TFCWNMF::TFCWNMF(int missing_count)
    : TFCRecom(missing_count),
      Recom(missing_count),
      user_factors_(),
      item_factors_(),
      prev_user_factors_(),
      prev_item_factors_(),
      observation_indicator_(),
      transposed_observation_indicator_(),
      transposed_sparse_missing_data_() {
    method_name_ = "TFCWNMF";
}

void TFCWNMF::set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda) {
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
    fuzzifier_Lambda_ = fuzzifier_Lambda;
    parameters_ = {(double)cluster_size_, fuzzifier_em_, fuzzifier_Lambda_, (double)latent_dimension_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
    user_factors_ = Tensor(cluster_size_, rs::num_users, latent_dimension_);
    item_factors_ = Tensor(cluster_size_, rs::num_items, latent_dimension_);
}

void TFCWNMF::set_initial_values(int seed) {
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
            membership_(i,k) = tmp_Mem[i];
        }
    }
    observation_indicator_ = sparse_missing_data_;
    for (int i = 0; i < user_factors_.rows(); i++) {
        for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
            if (observation_indicator_(i, j) != 0) {
                observation_indicator_(i, j) = 1.0;
            }
        }
    }
    transposed_observation_indicator_ = observation_indicator_.transpose();
    transposed_sparse_missing_data_ = sparse_missing_data_.transpose();
}

void TFCWNMF::calculate_factors() {
    prev_item_factors_ = item_factors_;
    prev_user_factors_ = user_factors_;

    Matrix item_numerator;
    Matrix item_denominator;
    SparseMatrix transposed_users_dot_items = transposed_sparse_missing_data_;

    for (int c = 0; c < cluster_size_; c++) {
        double tmp_diagonalMembership[rs::num_users];
        for (int i = 0; i < rs::num_users; i++) {
            tmp_diagonalMembership[i] = pow(membership_[c][i], fuzzifier_em_);
        }
        SparseMatrix diagonal_membership(rs::num_users, tmp_diagonalMembership, "diag");
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    transposed_users_dot_items(j, i) = user_factors_[c][i] * item_factors_[c][sparse_missing_data_(i, j, "index")];
                }
            }
        }
        Matrix membership_dot_user_factor = diagonal_membership * user_factors_[c];
        item_numerator = transposed_sparse_missing_data_ * membership_dot_user_factor;
        item_denominator = transposed_users_dot_items * membership_dot_user_factor;
        for (int row = 0; row < item_factors_.rows(); row++) {
            for (int col = 0; col < item_factors_.cols(); col++) {
                if (item_denominator(row, col) != 0) {
                    item_factors_[c](row, col) *= item_numerator(row, col) / item_denominator(row, col);
                }
            }
        }
    }

    Matrix user_numerator;
    Matrix user_denominator;
    SparseMatrix users_dot_items = sparse_missing_data_;

    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    users_dot_items(i, j) = user_factors_[c][i] * item_factors_[c][sparse_missing_data_(i, j, "index")];
                }
            }
        }
        user_numerator = sparse_missing_data_ * item_factors_[c];
        user_denominator = users_dot_items * item_factors_[c];
        for (int row = 0; row < user_factors_.rows(); row++) {
            for (int col = 0; col < user_factors_.cols(); col++) {
                if (user_denominator(row, col) != 0) {
                    user_factors_[c](row, col) *= user_numerator(row, col) / user_denominator(row, col);
                }
            }
        }
    }

    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            dissimilarities_(c, i) = 0.0;
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    double tmp = 0.0;
                    tmp = (sparse_missing_data_(i, j) - user_factors_[c][i] * item_factors_[c][sparse_missing_data_(i, j, "index")]);
                    dissimilarities_(c, i) = tmp * tmp;
                }
            }
        }
    }
    calculate_membership();
}

double TFCWNMF::calculate_objective_value() {
    double result;
    double user_factors__L2Norm, item_factors__L2Norm;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            result += pow(membership_[c][i], fuzzifier_em_) * dissimilarities_[c][i];
            +1 / (fuzzifier_Lambda_ * (fuzzifier_em_ - 1)) * (pow(membership_[c][i], fuzzifier_em_) - 1);
        }
    }
    return result;
}

bool TFCWNMF::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff = frobenius_norm(prev_user_factors_ - user_factors_) + frobenius_norm(prev_item_factors_ - item_factors_) +
                  frobenius_norm(prev_membership_ - membership_);
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

void TFCWNMF::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        prediction_[index] = 0.0;
        for (int c = 0; c < cluster_size_; c++) {
            prediction_[index] += membership_[c][missing_data_indices_[index][0]] *
                                  (user_factors_[c][missing_data_indices_[index][0]] * item_factors_[c][missing_data_indices_[index][1]]);
        }
        std::cout << "Prediction:" << prediction_[index]
                  << " SparseCorrectData:" << sparse_correct_data_(missing_data_indices_[index][0], missing_data_indices_[index][1]) << std::endl;
    }
}
