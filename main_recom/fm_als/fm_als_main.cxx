#include "../../src/recom_methods/fm_als/fm_als.h"

int main() {
    // 時間計測
    auto start = std::chrono::high_resolution_clock::now();

    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        FMWithALS recom(mv);
        recom.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                recom.set_parameters(ld, rp);
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

    // 計測終了
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int minutes = duration.count() / 60000;
    int seconds = (duration.count() % 60000) / 1000;
    int milliseconds = duration.count() % 1000;

    std::cout << "処理にかかった時間: " << minutes << " 分 " << seconds << " 秒 " << milliseconds << " ミリ秒" << std::endl;

    return 0;
}