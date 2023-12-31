#include "../../recom_system_base/recom.h"

class WNMF : virtual public Recom {
   protected:
    // 潜在次元
    int latent_dimension_;
    double reg_parameter_;
    // ユーザー行列とアイテム行列
    Matrix user_factors_, item_factors_;
    Matrix prev_user_factors_, prev_item_factors_;
    double *user_factor_values_,*item_factor_values_;
    //計算用
    SparseMatrix transpose_sparse_missing_data_;
    SparseMatrix sparse_prediction_;

   public:
    WNMF(int missing_count);
    void set_parameters(double latent_dimension_percentage, double reg_parameter);
    void calculate_factors() override;
    void set_initial_values(int seed) override;
    double calculate_objective_value() override;
    bool calculate_convergence_criterion() override;
    void calculate_prediction() override;
};