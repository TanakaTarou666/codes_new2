#include "../src/recom_methods/mf.h"

int main() {
    double latent_dimensions[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double reg_parameters[] = {0.01, 0.05, 0.09, 0.13};
    double learning_rates[] = {0.001};

    // 時間計測
    auto start = std::chrono::high_resolution_clock::now();

    for (int mv = start_missing_valu; mv <= end_missing_valu; mv += step_missing_valu) {
        MF recom(mv);
        recom.input(input_data_name);
        for (double ld : latent_dimensions) {
            for (double rp : reg_parameters) {
                for (double lr : learning_rates) {
                    recom.set_parameters(ld, lr, rp);
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

    // 計測終了
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int minutes = duration.count() / 60000;
    int seconds = (duration.count() % 60000) / 1000;
    int milliseconds = duration.count() % 1000;

    std::cout << "処理にかかった時間: " << minutes << " 分 " << seconds << " 秒 " << milliseconds << " ミリ秒" << std::endl;

    return 0;
}