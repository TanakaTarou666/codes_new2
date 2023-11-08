#include "../src/recom_methods/fm_als.h"

int main() {
    for (int mv = start_missing_valu; mv <= end_missing_valu; mv += step_missing_valu) {
        FMWithALS recom(mv);
        recom.input(input_data_name);
        recom.set_parameters(5, 0.001);
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
    return 0;
}