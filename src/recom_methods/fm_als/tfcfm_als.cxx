#include "tfcfm_als.h"

TFCFMWithALS::TFCFMWithALS(int missing_count)
    : FMBase(missing_count), TFCRecom(missing_count), Recom(missing_count), w0_(), prev_w0_(), w_(), prev_w_(), v_(), prev_v_(), e_(), q_(), x_() {
    method_name_ = "TFCFM_ALS";
}

void TFCFMWithALS::set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda) {
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
    parameters_ = {(double)latent_dimension_, double(cluster_size), fuzzifier_em, fuzzifier_Lambda};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
}

void TFCFMWithALS::set_initial_values(int seed) {
    seed *= 1000000;
    w0_ = Vector(cluster_size_, 0.0, "all");
    w_ = Matrix(cluster_size_, rs::num_users + rs::num_items, 0.0);
    v_ = Tensor(cluster_size_, rs::num_users + rs::num_items, latent_dimension_);
    e_ = Matrix(cluster_size_, sparse_missing_data_.nnz() - num_missing_value_, 0.0);
    q_ = Tensor(cluster_size_, sparse_missing_data_.nnz() - num_missing_value_, latent_dimension_);
    x_ = DSSTensor(sparse_missing_data_, rs::num_users + rs::num_items);
    membership_ = Matrix(cluster_size_, rs::num_users, 1.0 / (double)cluster_size_);
    dissimilarities_ = Matrix(cluster_size_, rs::num_users, 0);

    std::mt19937_64 mt;
    for (int c = 0; c < cluster_size_; c++) {
        for (int n = 0; n < rs::num_users + rs::num_items; n++) {
            for (int k = 0; k < latent_dimension_; k++) {
                mt.seed(seed);
                // ランダムに値生成
                std::uniform_real_distribution<> rand_v(0.0, 0.001);
                v_[c](n, k) = rand_v(mt);
            }
        }
    }

    for (int i = 0; i < rs::num_users; i++) {
        for (int j = 0; j < x_(i, "row"); j++) {
            SparseVector x_element(rs::num_items, 2);
            x_element(0) = 1;
            x_element(0, "index") = i;
            x_element(1) = 1;
            x_element(1, "index") = rs::num_users + x_(i, j, "index");
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
            membership_(i,k) = tmp_Mem[i];
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

void TFCFMWithALS::precompute() {
    for (int c = 0; c < cluster_size_; ++c) {
        int l = 0;
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    e_(c, l) = predict_y(x_(i, j), w0_[c], w_[c], v_[c]) - sparse_missing_data_(i, j);
                    for (int f = 0; f < latent_dimension_; ++f) {
                        q_[c](l, f) = 0.0;
                        for (int j_ = 0; j_ < x_(i, j).nnz(); ++j_) {
                            q_[c](l, f) += x_(i, j)(j_) * v_[c](x_(i, j)(j_, "index"), f);
                        }
                    }
                    l++;
                }
            }
        }
    }
}

void TFCFMWithALS::calculate_factors() {
    prev_v_ = v_;
    prev_w_ = w_;
    prev_w0_ = w0_;
    prev_membership_ = membership_;
    double sum_e = 0;
    for (int c = 0; c < cluster_size_; ++c) {
        double numerator = 0;
        double denominator = 0;
        int l = 0;
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    numerator += pow(membership_(c, i), fuzzifier_em_) * (e_(c, l) - w0_[c]);
                    denominator += pow(membership_(c, i), fuzzifier_em_);
                    l++;
                }
            }
        }
        double w0a = -numerator / denominator;
        for (l = 0; l < e_.cols(); l++) e_(c, l) += (w0a - w0_[c]);
        w0_[c] = w0a;
    }

    // 1-way interactions
    for (int c = 0; c < cluster_size_; ++c) {
        double wa[w_.cols()] = {};
        for (int a = 0; a < 2; ++a) {
            double numerator_w[w_.cols()] = {};
            double denominator_w[w_.cols()] = {};
            int l = 0;
            for (int i = 0; i < sparse_missing_data_.rows(); i++) {
                for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                    if (sparse_missing_data_(i, j) != 0) {
                        numerator_w[x_(i, j)(a, "index")] +=
                            pow(membership_(c, i), fuzzifier_em_) * (e_(c, l) - w_(c, x_(i, j)(a, "index")) * x_(i, j)(a)) * x_(i, j)(a);
                        denominator_w[x_(i, j)(a, "index")] += pow(membership_(c, i), fuzzifier_em_) * x_(i, j)(a) * x_(i, j)(a);
                    }
                }
            }
            for (int i = 0; i < w_.cols(); ++i) {
                if (denominator_w[i] != 0 && std::isfinite(denominator_w[i])) wa[i] = -numerator_w[i] / denominator_w[i];
            }
            l = 0;
            for (int i = 0; i < sparse_missing_data_.rows(); i++) {
                for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                    if (sparse_missing_data_(i, j) != 0) {
                        e_(c, l) += (wa[x_(i, j)(a, "index")] - w_(c, x_(i, j)(a, "index"))) * x_(i, j)(a);
                        l++;
                    }
                }
            }
        }
        for (int a = 0; a < w_.cols(); ++a) {
            w_(c, a) = wa[a];
        }
    }

    // 2-way interactions
    for (int c = 0; c < cluster_size_; ++c) {
        for (int f = 0; f < latent_dimension_; ++f) {
            double va[v_.rows()] = {};
            for (int a = 0; a < 2; ++a) {
                double h_value[e_.cols()] = {};
                int l = 0;
                for (int i = 0; i < sparse_missing_data_.rows(); i++) {
                    for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                        if (sparse_missing_data_(i, j) != 0) {
                            h_value[l] = -x_(i, j)(a) * (x_(i, j)(a) * v_[c](x_(i, j)(a, "index"), f) - q_[c](l, f));
                            l++;
                        }
                    }
                }
                double numerator_v[v_.rows()] = {};
                double denominator_v[v_.rows()] = {};
                l = 0;
                for (int i = 0; i < sparse_missing_data_.rows(); i++) {
                    for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                        if (sparse_missing_data_(i, j) != 0) {
                            numerator_v[x_(i, j)(a, "index")] +=
                                pow(membership_(c, i), fuzzifier_em_) * (e_(c, l) - v_[c](x_(i, j)(a, "index"), f) * h_value[l]) * h_value[l];
                            denominator_v[x_(i, j)(a, "index")] += pow(membership_(c, i), fuzzifier_em_) * h_value[l] * h_value[l];
                            l++;
                        }
                    }
                }
                for (int a = 0; a < v_.rows(); ++a) {
                    if (denominator_v[a] != 0 && std::isfinite(denominator_v[a])) va[a] = -numerator_v[a] / denominator_v[a];
                }
                l = 0;
                for (int i = 0; i < sparse_missing_data_.rows(); i++) {
                    for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                        if (sparse_missing_data_(i, j) != 0) {
                            e_(c, l) += (va[x_(i, j)(a, "index")] - v_[c](x_(i, j)(a, "index"), f)) * h_value[l];
                            q_[c](l, f) += (va[x_(i, j)(a, "index")] - v_[c](x_(i, j)(a, "index"), f)) * x_(i, j)(a);
                            l++;
                        }
                    }
                }
            }
            for (int a = 0; a < v_.rows(); ++a) {
                v_[c](a, f) = va[a];
            }
        }
    }

    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < sparse_missing_data_.rows(); i++) {
            dissimilarities_(c, i) = 0.0;
            for (int j = 0; j < sparse_missing_data_(i, "row"); j++) {
                if (sparse_missing_data_(i, j) != 0) {
                    double tmp = 0.0;
                    tmp = (sparse_missing_data_(i, j) - predict_y(x_(i, j), w0_[c], w_[c], v_[c]));
                    dissimilarities_(c, i) = tmp * tmp;
                }
            }
        }
    }
    calculate_membership();
}

double TFCFMWithALS::calculate_objective_value() {
    double result;
    for (int c = 0; c < cluster_size_; c++) {
        for (int i = 0; i < rs::num_users; i++) {
            result += pow(membership_[c][i], fuzzifier_em_) * dissimilarities_[c][i];
            +1 / (fuzzifier_Lambda_ * (fuzzifier_em_ - 1)) * (pow(membership_[c][i], fuzzifier_em_) - 1);
        }
    }
    return result;
}

bool TFCFMWithALS::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff =
        squared_norm(prev_w0_ - w0_) + frobenius_norm(prev_w_ - w_) + frobenius_norm(prev_v_ - v_) + frobenius_norm(prev_membership_ - membership_);
    // std::cout << "w0:" << squared_norm(prev_w0_ - w0_) << std::endl;
    // std::cout << "w:" << frobenius_norm(prev_w_ - w_) << std::endl;
    // std::cout << "v:" << frobenius_norm(prev_v_ - v_) << std::endl;
    // std::cout << "m:" << frobenius_norm(prev_membership_ - membership_) << std::endl;
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
            prediction_[index] += membership_[c][missing_data_indices_[index][0]] *
                                  predict_y(x_(missing_data_indices_[index][0], missing_data_indices_[index][1]), w0_[c], w_[c], v_[c]);
        }
        // std::cout << "Prediction:" << prediction_[index]
        //           << " SparseCorrectData:" << sparse_correct_data_(missing_data_indices_[index][0], missing_data_indices_[index][1]) << std::endl;
    }
}
