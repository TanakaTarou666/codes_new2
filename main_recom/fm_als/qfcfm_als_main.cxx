#include "../../src/recom_methods/fm_als/qfcfm_als.h"

int main(int argc, char *argv[]) {
    int start_latent_dimension = std::stoi(argv[1]);
    int end_latent_dimension = std::stoi(argv[2]);
    if (check_command_args(argc, argv)) {
        exit(1);
    }
    // 時間計測
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        QFCFMWithALS recom(mv);
        recom.input(rs::input_data_name);
        for (int ld = start_latent_dimension; ld <= end_latent_dimension; ld++) {
            for (double rp : rs::reg_parameters) {
                for (int c : rs::cluster_size) {
                    for (double em : rs::fuzzifier_em) {
                        for (double lambda : rs::fuzzifier_lambda) {
                            recom.set_parameters(rs::latent_dimensions[ld], c, em, lambda, rp);
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

    // 計測終了
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int minutes = duration.count() / 60000;
    int seconds = (duration.count() % 60000) / 1000;
    int milliseconds = duration.count() % 1000;

    std::cout << "処理にかかった時間: " << minutes << " 分 " << seconds << " 秒 " << milliseconds << " ミリ秒" << std::endl;

    return 0;
}