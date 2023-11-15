#include "../../src/recom_methods/mf/qfcmf.h"

int main() {
    // 時間計測
    auto start = std::chrono::high_resolution_clock::now();

    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        QFCMF recom(mv);
        recom.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                for (double lr : rs::learning_rates) {
                    for (int c : rs::cluster_size) {
                        for (double em : rs::fuzzifier_em) {
                            for (double lambda : rs::fuzzifier_lambda) {
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

    // 計測終了 
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "処理にかかった時間: " << duration.count() / 60000 << " 分 " << (duration.count() % 60000) / 1000 << " 秒 "
              << duration.count() % 1000 << " ミリ秒" << std::endl;

    return 0;
}