#include "../src/recom_methods/tfcfm_als.h"

int main() {
    for (int mv = start_missing_valu; mv <= end_missing_valu; mv += step_missing_valu) {
        TFCFMWithALS recom(mv);
        recom.input(input_data_name);
        recom.set_parameters(5, 2, 1.01, 1000);
        for (int i = 0; i < missing_pattern; i++) {
            recom.revise_missing_values();
            recom.train();
            recom.calculate_mae(i);
            recom.calculate_roc(i);
        }
        recom.precision_summury();
    }
    return 0;
}