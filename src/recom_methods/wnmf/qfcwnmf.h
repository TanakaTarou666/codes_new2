#include "../../recom_system_base/qfc_recom.h"

class QFCWNMF : virtual public QFCRecom {

   protected:
    // 潜在次元
    int latent_dimension_;
    // ユーザー行列とアイテム行列
    Tensor user_factors_, item_factors_;
    Tensor prev_user_factors_, prev_item_factors_;
    double *user_factor_values_,*item_factor_values_;
    //計算用
    SparseMatrix transpose_sparse_missing_data_;
    SparseMatrix sparse_prediction_;
    Matrix tmp_user_factors_, tmp_item_factors_;
    SparseMatrix tmp_membership_;

   public:
    QFCWNMF(int missing_pattern);
    void set_parameters(double latent_dimension_percentage, int cluster_size, double fuzzifier_em, double fuzzifier_Lambda);
    void calculate_factors() override;
    void set_initial_values(int seed) override;
    double calculate_objective_value() override;
    bool calculate_convergence_criterion() override;
    void calculate_prediction() override;
};