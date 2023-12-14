#include "fm_als.h"

FMWithALS::FMWithALS(int missing_count)
    : FMBase(missing_count), Recom(missing_count), w0_(), prev_w0_(), w_(), prev_w_(), v_(), prev_v_(), e_(), q_(), x_() {
    method_name_ = append_current_time_if_test("FM_ALS");
}

void FMWithALS::set_parameters(double latent_dimension_percentage, double reg_parameter) {
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
    parameters_ = {(double)latent_dimension_, reg_parameter_};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
}

void FMWithALS::set_initial_values(int seed) {
    seed *= 1000000;
    w0_ = 0.0;
    w_ = Vector(sum_users_items, 0.0, "all");
    v_ = Matrix(latent_dimension_, sum_users_items);
    e_ = Vector(sparse_missing_data_.nnz(), 0.0, "all");
    q_ = Matrix(latent_dimension_, sparse_missing_data_.nnz());
    x_ = DSSTensor(sparse_missing_data_, sum_users_items);

    std::mt19937_64 mt;
    for (int n = 0; n < sum_users_items; n++) {
        for (int k = 0; k < latent_dimension_; k++) {
            mt.seed(seed);
            // ランダムに値生成
            std::uniform_real_distribution<> rand_v(-0.01, 0.01);
            v_(k, n) = rand_v(mt);
            // v_(n, k) = 0.01 * k + 0.01 * n;
            // v_(n, k) = 1.0;
            seed++;
        }
    }

    for (int i = 0; i < rs::num_users; i++) {
        for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
            x_(i, j) = make_one_hot_data(i, x_.dense_index(i, j));
        }
    }


    // データ表示
    // for (int i = 0; i < rs::num_users; i++) {
    //     for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
    //         std::cout << "i:" << i << " j:" << j << " : " << x_(i, j)
    //                   << std::endl;
    //     }
    // }
    precompute();
}

void FMWithALS::precompute() {
    int l = 0;
    SparseVector tmp_x;
    for (int i = 0; i < rs::num_users; i++) {
        for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
            tmp_x = x_(i, j);
            e_[l] = predict_y(tmp_x, w0_, w_, v_) - sparse_missing_data_(i, j);
            for (int f = 0; f < latent_dimension_; ++f) {
                q_(f, l) = 0.0;
                for (int j_ = 0; j_ < tmp_x.nnz(); ++j_) {
                    q_(f, l) += tmp_x(j_) * v_(f, tmp_x.dense_index(j_));
                }
            }
            l++;
        }
        
    }
}
void FMWithALS::calculate_factors() {
    prev_v_ = v_;
    prev_w_ = w_;
    prev_w0_ = w0_;
    SparseVector tmp_x;
    double numerator_w0 = 0;
    int l = 0;
    for (int i = 0; i < rs::num_users; i++) {
        for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
            numerator_w0 += (e_[l] - w0_);
            l++;
        }
    }
    double w0a = -numerator_w0 / (l + reg_parameter_);
    for (l = 0; l < e_.size(); l++) e_[l] += (w0a - w0_);
    w0_ = w0a;

    // 1-way interactions
    double wa[sum_users_items] = {};
    for (int a = 0; a < 2; ++a) {
        double numerator_w[sum_users_items] = {};
        double denominator_w[sum_users_items] = {};
        l = 0;
        for (int i = 0; i < rs::num_users; i++) {
            for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                tmp_x = x_(i, j);
                double tmp_x_value = tmp_x(a);
                int tmp_x_dense_index = tmp_x.dense_index(a);
                numerator_w[tmp_x_dense_index] += (e_[l] - w_[tmp_x_dense_index] * tmp_x_value) * tmp_x_value;
                denominator_w[tmp_x_dense_index] += tmp_x_value * tmp_x_value;
                l++;
            }
        }
        for (int i = 0; i < sum_users_items; ++i) {
            if (denominator_w[i] != 0 && std::isfinite(denominator_w[i])) wa[i] = -numerator_w[i] / (denominator_w[i] + reg_parameter_);
        }
        l = 0;
        for (int i = 0; i < rs::num_users; i++) {
            for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                tmp_x = x_(i, j);
                double tmp_x_value = tmp_x(a);
                int tmp_x_dense_index = tmp_x.dense_index(a);
                e_[l] += (wa[tmp_x_dense_index] - w_[tmp_x_dense_index]) * tmp_x_value;
                l++;
            }
        }
    }
    for (int a = 0; a < sum_users_items; ++a) {
        w_[a] = wa[a];
    }

    // 2-way interactions
    for (int f = 0; f < latent_dimension_; ++f) {
        double* tmp_v = v_[f].get_values();
        double* tmp_q = q_[f].get_values();
        double va[sum_users_items] = {};
        for (int a = 0; a < 2; ++a) {
            double h_value[e_.size()] = {};
            l = 0;
            for (int i = 0; i < rs::num_users; i++) {
                for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                    tmp_x = x_(i, j);
                    double tmp_x_value = tmp_x(a);
                    int tmp_x_dense_index = tmp_x.dense_index(a);
                    h_value[l] = -tmp_x_value * (tmp_x_value * tmp_v[tmp_x_dense_index] - tmp_q[l]);
                    l++;
                }
            }

            double numerator_v[sum_users_items] = {};
            double denominator_v[sum_users_items] = {};
            l = 0;
            for (int i = 0; i < rs::num_users; i++) {
                for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                    tmp_x = x_(i, j);
                    double tmp_x_value = tmp_x(a);
                    int tmp_x_dense_index = tmp_x.dense_index(a);
                    double tmp_h = h_value[l];
                    numerator_v[tmp_x_dense_index] += (e_[l] - tmp_v[tmp_x_dense_index] * tmp_h) * tmp_h;
                    denominator_v[tmp_x_dense_index] += tmp_h * tmp_h;
                    l++;
                }
            }
            for (int a = 0; a < sum_users_items; ++a) {
                if (denominator_v[a] != 0 && std::isfinite(denominator_v[a])) va[a] = -numerator_v[a] / (denominator_v[a] + reg_parameter_);
            }
            l = 0;
            for (int i = 0; i < rs::num_users; i++) {
                for (int j = 0; j < sparse_missing_data_row_nnzs_[i]; j++) {
                    tmp_x = x_(i, j);
                    double tmp_x_value = tmp_x(a);
                    int tmp_x_dense_index = tmp_x.dense_index(a);
                    e_[l] += (va[tmp_x_dense_index] - tmp_v[tmp_x_dense_index]) * h_value[l];
                    q_(f, l) += (va[tmp_x_dense_index] - tmp_v[tmp_x_dense_index]) * tmp_x_value;
                    l++;
                }
            }
        }
        for (int a = 0; a < sum_users_items; ++a) {
            v_(f, a) = va[a];
        }
    }
}

double FMWithALS::calculate_objective_value() {
    double result = 0.0;
    for (int i = 0; i < sparse_missing_data_.rows(); i++) {
        for (int j = 0; j < sparse_missing_data_.nnz(i); j++) {
            if (sparse_missing_data_(i, j) != 0) {
                double tmp = 0.0;
                tmp = (sparse_missing_data_(i, j) - predict_y(x_(i, j), w0_, w_, v_));
                result += tmp * tmp;
            }
        }
    }
    result += reg_parameter_ * (fabs(w0_) + squared_sum(w_) + squared_sum(v_))/2;
    return result;
}

bool FMWithALS::calculate_convergence_criterion() {
    bool result = false;
#if defined ARTIFICIALITY
    double diff = abs(prev_w0_ - w0_) + squared_norm(prev_w_ - w_) + frobenius_norm(prev_v_ - v_);
    std::cout << " diff:" << diff << " L:" << calculate_objective_value() << "\t";
    std::cout << "w0:" << abs(prev_w0_ - w0_) << "\t";
    std::cout << "w:" << squared_norm(prev_w_ - w_) << "\t";
    std::cout << "v:" << frobenius_norm(prev_v_ - v_) << "\t";
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

void FMWithALS::calculate_prediction() {
    for (int index = 0; index < num_missing_value_; index++) {
        prediction_[index] = 0.0;
        SparseVector tmp = make_one_hot_data(missing_data_indices_(index, 0), sparse_missing_data_cols_[index]);
        prediction_[index] += predict_y(tmp, w0_, w_, v_);
        // std::cout << "Prediction:" << prediction_[index]
        //           << " SparseCorrectData:"
        //           << sparse_correct_data_(missing_data_indices_(index,0),
        //                                   missing_data_indices_(index,1))
        //           << std::endl;
    }
}