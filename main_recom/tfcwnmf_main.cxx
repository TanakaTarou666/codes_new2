#include "../src/recom_methods/tfcwnmf.h"

int main() {
    double latent_dimensions[] = {5.0};
    int cluster_size[] = {5};
    double fuzzifier_em[] = {1.001};
    double fuzzifier_Lambda[] = {1000};

    for (int mv = start_missing_valu; mv <= end_missing_valu; mv += step_missing_valu) {
        TFCWNMF recom(mv);
        recom.input(input_data_name);
        for (double ld : latent_dimensions) {
            for (int c : cluster_size) {
                for (double em : fuzzifier_em) {
                    for (double lambda : fuzzifier_Lambda) {
                        recom.set_parameters(ld, c, em, lambda);
                    }
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

    return 0;
}