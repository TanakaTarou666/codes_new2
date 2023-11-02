#include "../src/recom_methods/tfcmf.h"

int main() {
    double latent_dimensions[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double reg_parameters[] = {0.01, 0.05, 0.09, 0.13};
    double learning_rates[] = {0.001};
    int cluster_size[] = {2, 4, 5, 8, 9};
    double fuzzifier_em[] = {1.001};
    double fuzzifier_Lambda[] = {1000};

    for (int mv = start_missing_valu; mv <= end_missing_valu; mv += step_missing_valu) {
        TFCMF recom(mv);
        recom.input(input_data_name);
        for (double ld : latent_dimensions) {
            for (double rp : reg_parameters) {
                for (double lr : learning_rates) {
                    for (int c : cluster_size) {
                        for (double em : fuzzifier_em) {
                            for (double lambda : fuzzifier_Lambda) {
                                recom.set_parameters(ld, c, em, lambda, rp, lr);
                                for (int i = 0; i < missing_pattern; i++) {
                                    // データを欠損
                                    recom.revise_missing_values();
                                    recom.train();
                                    recom.calculate_mae(i);
                                    recom.calculate_roc(i);
                                }
                                // 指標値の計算 シード値のリセット
                                recom.precision_summury();
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}