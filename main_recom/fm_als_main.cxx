#include "../src/recom_methods/fm_als.h"

int main() {
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        FMWithALS recom(mv);
        recom.input(rs::input_data_name);
        recom.set_parameters(5, 0.001);
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
    return 0;
}