#include "../src/recom_methods/fm_als.h"

int main() {
    FMWithALS recom(0);
    recom.input(input_data_name);
    recom.set_parameters(5, 0.001);
    recom.revise_missing_values();
    recom.train();
    return 0;
}