#include "tfcfm_sgd.h"

TFCFMWithSGD::TFCFMWithSGD(int missing_count)
    : TFCRecom(missing_count), Recom(missing_count), w0_(), prev_w0_(), w_(), prev_w_(), v_(), prev_v_(), e_(), q_(), X_() {
    method_name_ = "TFCFM_SGD";
}

void TFCFMWithSGD::set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda, double reg_parameter,double learning_rate) {
#if defined ARTIFICIALITY
    latent_dimension_ = latent_dimension_percentage;
#elif
    if (num_users > num_items) {
        latent_dimension_ = std::round(num_items * latent_dimension_percentage / 100);
    } else {
        latent_dimension_ = std::round(num_users * latent_dimension_percentage / 100);
    }
    if (steps < 50) {
        std::cerr << "MF: \"step\" should be 50 or more.";
        return 1;
    }
#endif
    cluster_size_ = cluster_size;
    fuzzifier_em_ = fuzzifier_em;
    fuzzifier_Lambda_ = fuzzifier_Lambda;
    reg_parameter_ = reg_parameter;
    learning_rate_ = learning_rate;
    parameters_ = {(double)latent_dimension_, double(cluster_size), fuzzifier_em, fuzzifier_Lambda, double learning_rate, double reg_parameter};
    dirs_ = mkdir_result({method_name_}, parameters_, num_missing_value_);
}

void TFCFMWithSGD::train() {
    int error_count = 0;
    double best_objective_value = DBL_MAX;
    for (int initial_value_index = 0; initial_value_index < num_initial_values; initial_value_index++) {
        std::cout << method_name_ << ": initial setting " << initial_value_index << std::endl;
        set_initial_values(initial_value_index);
        error_detected_ = false;
#ifndef ARTIFICIALITY
        prev_objective_value_ = DBL_MAX;
#endif
        // for (int step = 0; step < steps; step++) {
        //     calculate_Wo_w_v();
        //     // 収束条件
        //     if (calculate_convergence_criterion()) {
        //         break;
        //     }
        //     if (step == steps - 1) {
        //         error_detected_ = true;
        //         break;
        //     }
        // }

        // if (error_detected_) {
        //     error_count++;
        //     // 初期値全部{NaN出た or step上限回更新して収束しなかった} => 1を返して終了
        //     if (error_count == num_initial_values) {
        //         return;
        //     }
        // } else {
        //     double objective_value = calculate_objective_value();
        //     if (objective_value < best_objective_value) {
        //         best_objective_value = objective_value;
        //         calculate_prediction();
        //     }
        // }
    }
}

void TFCFMWithSGD::set_initial_values(int &seed) {
    w0_ = Vector(cluster_size_, 0.0, "all");
    w_ = Matrix(cluster_size_, num_users + num_items, 0.0);
    v_ = Tensor(cluster_size_, num_users + num_items, latent_dimension_);
    X_ = DSDTensor(sparse_missing_data_, num_users + num_items);

    std::mt19937_64 mt;
    for (int c = 0; c < cluster_size_; c++) {
        for (int n = 0; n < num_users + num_items; n++) {
            for (int k = 0; k < latent_dimension_; k++) {
                mt.seed(seed);
                // ランダムに値生成
                std::uniform_real_distribution<> rand_v(0.001, 1.0);
                v_[c](n, k);
            }
        }
    }

    for (int i = 0; i < num_users; i++) {
        for (int j = 0; j < num_items; j++) {
            X_(i, j)[i] = 1;
            X_(i, j)[num_users + j] = 1;
        }
    }

    // データ表示
    // for (int i = 0; i < 1; i++) {
    //     for (int j = 0; j < num_items; j++) {
    //         std::cout << 100 * i + j << std::endl;
    //         X_(i, j).print_values();
    //     }
    // }
}

void TFCFMWithSGD::calculate_Wo_w_v() {}

double TFCFMWithSGD::calculate_objective_value() {
    double result = 0.0;
    
    for (int c = 0; c < cluster_size_; c++) {
        double reg_w = 0,reg_v=0;
        for (int i = 0; i < num_users; i++) {
            result += pow(membership_(i,j),fuzzifier_em_)*dissimilarities_(c,i)
                      + 1/(fuzzifier_Lambda_*(fuzzifier_em_-1))*(pow(membership_(i,j)),fuzzifier_em_)-membership_(i,j);
        }
        for (int n = 0; n < num_users + num_items; n++) {
            reg_w += pow(w_(c,n), 2);
            for (int k = 0; k < latent_dimension_; k++) {
                reg_v += pow(v_[c](n, k), 2);
            }
        }
        result += reg_parameter_*(pow(w0_[c], 2) + reg_w + reg_v);
    }
    return result;
}


