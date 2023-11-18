#include "../../src/recom_methods/wnmf/wnmf.h"

int main() {
    // 時間計測
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        WNMF recom(mv);
        recom.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            recom.set_parameters(ld);
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

    // 計測終了
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "処理にかかった時間: " << duration.count() / 60000 << " 分 " << (duration.count() % 60000) / 1000 << " 秒 "
              << duration.count() % 1000 << " ミリ秒" << std::endl;

    return 0;
}