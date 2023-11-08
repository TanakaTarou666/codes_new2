#include "../src/recom_methods/tfcmf.h"

int main() {
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        TFCMF recom(mv);
        recom.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                for (double lr : rs::learning_rates) {
                    for (int c : rs::cluster_size) {
                        for (double em : rs::fuzzifier_em) {
                            for (double lambda : rs::fuzzifier_Lambda) {
                                recom.set_parameters(ld, c, em, lambda, rp, lr);
                                for (int i = 0; i < rs::missing_pattern; i++) {
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