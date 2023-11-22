#include "../../src/recom_methods/fm_als/fm_als.h"
#include "../../src/recom_methods/fm_sgd/fm_sgd.h"
#include "../../src/recom_methods/mf/mf.h"
#include "../../src/recom_methods/mf/qfcmf.h"
#include "../../src/recom_methods/mf/tfcmf.h"

int main() {
    // //MF
    // for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
    //     MF mf(mv);
    //     mf.input(rs::input_data_name);
    //     for (double ld : rs::latent_dimensions) {
    //         for (double rp : rs::reg_parameters) {
    //             for (double lr : rs::learning_rates) {
    //                 mf.set_parameters(ld, lr, rp);
    //                 mf.tally_result();
    //             }
    //         }
    //     }
    //     mf.output_high_score_in_tally_result();
    // }
    // for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
    //     TFCMF tfcmf(mv);
    //     QFCMF qfcmf(mv);
    //     tfcmf.input(rs::input_data_name);
    //     for (double ld : rs::latent_dimensions) {
    //         for (double rp : rs::reg_parameters) {
    //             for (double lr : rs::learning_rates) {
    //                 for (int c : rs::cluster_size) {
    //                     for (double em : rs::fuzzifier_em) {
    //                         for (double lambda : rs::fuzzifier_lambda) {
    //                             tfcmf.set_parameters(ld, c, em, lambda, rp, lr);
    //                             tfcmf.tally_result();
    //                             qfcmf.set_parameters(ld, c, em, lambda, rp, lr);
    //                             qfcmf.tally_result();
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     tfcmf.output_high_score_in_tally_result();
    //     qfcmf.output_high_score_in_tally_result();
    // }

    // // FM_SGD
    // for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
    //     FMWithSGD fm_sgd(mv);
    //     fm_sgd.input(rs::input_data_name);
    //     for (double ld : rs::latent_dimensions) {
    //         for (double rp : rs::reg_parameters) {
    //             for (double lr : rs::learning_rates) {
    //                 fm_sgd.set_parameters(ld, rp, lr);
    //                 fm_sgd.tally_result();
    //             }
    //         }
    //     }
    //     fm_sgd.output_high_score_in_tally_result();
    // }

    //FM_ALS
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        FMWithALS fm_als(mv);
        fm_als.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                fm_als.set_parameters(ld, rp);
                fm_als.tally_result();
            }
        }
        fm_als.output_high_score_in_tally_result();
    }
    
    return 0;
}