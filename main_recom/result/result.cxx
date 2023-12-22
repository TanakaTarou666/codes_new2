#include "../../src/recom_methods/fm_als/fm_als.h"
#include "../../src/recom_methods/fm_als/qfcfm_als.h"
#include "../../src/recom_methods/fm_als/tfcfm_als.h"
#include "../../src/recom_methods/fm_sgd/fm_sgd.h"
#include "../../src/recom_methods/fm_sgd/tfcfm_sgd.h"
#include "../../src/recom_methods/mf/mf.h"
#include "../../src/recom_methods/mf/qfcmf.h"
#include "../../src/recom_methods/mf/tfcmf.h"
#include "../../src/recom_methods/wnmf/qfcwnmf.h"
#include "../../src/recom_methods/wnmf/tfcwnmf.h"
#include "../../src/recom_methods/wnmf/wnmf.h"
// #define __MF__
// #define __TFCMF__
// #define __QFCMF__
#define __QFCWNMF__
// #define __FM_SGD__
// #define __TFCFM_SGD__
// #define __FM_ALS__
// #define __TFCFM_ALS__
// #define __QFCFM_ALS__

int main() {
#if defined __MF__
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        MF mf(mv);
        mf.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                for (double lr : rs::learning_rates) {
                    mf.set_parameters(ld, lr, rp);
                    mf.tally_result();
                }
            }
        }
        mf.output_high_score_in_tally_result();
    }
#endif

#if defined __TFCMF__
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        TFCMF tfcmf(mv);
        tfcmf.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                for (double lr : rs::learning_rates) {
                    for (int c : rs::cluster_size) {
                        for (double em : rs::fuzzifier_em) {
                            for (double lambda : rs::fuzzifier_lambda) {
                                tfcmf.set_parameters(ld, c, em, lambda, rp, lr);
                                tfcmf.tally_result();
                            }
                        }
                    }
                }
            }
        }
        tfcmf.output_high_score_in_tally_result();
    }
#endif

#if defined __QFCMF__
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        QFCMF qfcmf(mv);
        qfcmf.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                for (double lr : rs::learning_rates) {
                    for (int c : rs::cluster_size) {
                        for (double em : rs::fuzzifier_em) {
                            for (double lambda : rs::fuzzifier_lambda) {
                                qfcmf.set_parameters(ld, c, em, lambda, rp, lr);
                                qfcmf.tally_result();
                            }
                        }
                    }
                }
            }
        }
        qfcmf.output_high_score_in_tally_result();
    }
#endif

#if defined __QFCWNMF__
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        QFCWNMF qfcwnmf(mv);
        qfcwnmf.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (int c : rs::cluster_size) {
                for (double em : rs::fuzzifier_em) {
                    for (double lambda : rs::fuzzifier_lambda) {
                        qfcwnmf.set_parameters(ld, c, em, lambda);
                        qfcwnmf.tally_result();
                    }
                }
            }
        }
        qfcwnmf.output_high_score_in_tally_result();
    }
#endif

#if defined __FM_SGD__
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        FMWithSGD fm_sgd(mv);
        fm_sgd.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                for (double lr : rs::learning_rates) {
                    fm_sgd.set_parameters(ld, rp, lr);
                    fm_sgd.tally_result();
                }
            }
        }
        fm_sgd.output_high_score_in_tally_result();
    }
#endif

#if defined __TFCFM_SGD__
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        TFCFMWithSGD recom(mv);
        recom.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                for (double lr : rs::learning_rates) {
                    for (int c : rs::cluster_size) {
                        for (double em : rs::fuzzifier_em) {
                            for (double lambda : rs::fuzzifier_lambda) {
                                recom.set_parameters(ld, c, em, lambda, rp, lr);
                                recom.tally_result();
                            }
                        }
                    }
                }
            }
        }
        recom.output_high_score_in_tally_result();
    }
#endif

#if defined __FM_ALS__
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
#endif

#if defined __TFCFM_ALS__
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        TFCFMWithALS tfcfm_als(mv);
        tfcfm_als.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                for (int c : rs::cluster_size) {
                    for (double em : rs::fuzzifier_em) {
                        for (double lambda : rs::fuzzifier_lambda) {
                            tfcfm_als.set_parameters(ld, c, em, lambda, rp);
                            tfcfm_als.tally_result();
                        }
                    }
                }
            }
        }
        tfcfm_als.output_high_score_in_tally_result();
    }
#endif

#if defined __QFCFM_ALS__
    for (int mv = rs::start_missing_valu; mv <= rs::end_missing_valu; mv += rs::step_missing_valu) {
        QFCFMWithALS qfcfm_als(mv);
        qfcfm_als.input(rs::input_data_name);
        for (double ld : rs::latent_dimensions) {
            for (double rp : rs::reg_parameters) {
                for (int c : rs::cluster_size) {
                    for (double em : rs::fuzzifier_em) {
                        for (double lambda : rs::fuzzifier_lambda) {
                            qfcfm_als.set_parameters(ld, c, em, lambda, rp);
                            qfcfm_als.tally_result();
                        }
                    }
                }
            }
        }
        qfcfm_als.output_high_score_in_tally_result();
    }
#endif

    return 0;
}
