#include "../../src/recom_methods/fm_sgd/tfcfm_sgd.h"

int main() {
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        TFCFMWithSGD recom(mv);
        recom.input(rs::input_data_name);
        recom.set_parameters(5, 2, 1.01, 1000,0.001,0.01);
        for (int i = 0; i < rs::missing_pattern; i++) {
            recom.revise_missing_values();
            recom.train();
            recom.calculate_mae(i);
            recom.calculate_roc(i);
        }
        recom.precision_summury();
    }
    return 0;
}