#include "sparse_matrix.h"

// コンストラクタ
SparseMatrix::SparseMatrix(int rows, int cols) : rows_(rows), cols_(cols) {
    // 行ポインタ、列インデックス、データを初期化
    row_pointers_ = new int[rows + 1]();
    col_indices_ = nullptr;  // サイズは後で設定する
    values_ = nullptr;       // サイズは後で設定する
    nnz_ = 0;
}
// コンストラクタ
SparseMatrix::SparseMatrix(int rows, int cols, int nnz) : rows_(rows), cols_(cols), nnz_(nnz) {
    // 行ポインタ、列インデックス、データを初期化
    row_pointers_ = new int[rows + 1]();
    col_indices_ = new int[nnz];
    values_ = new double[nnz];
}
// コンストラクタ
SparseMatrix::SparseMatrix() : rows_(0), cols_(0), nnz_(0) {
    // 行ポインタ、列インデックス、データを初期化
    row_pointers_ = nullptr;
    col_indices_ = nullptr;
    values_ = nullptr;
}
// コピーコンストラクタ
SparseMatrix::SparseMatrix(const SparseMatrix& arg) : rows_(arg.rows_), cols_(arg.cols_), nnz_(arg.nnz_) {
    // 行ポインタ、列インデックス、データをコピー
    row_pointers_ = new int[rows_ + 1];
    col_indices_ = new int[nnz_];
    values_ = new double[nnz_];

    for (int i = 0; i <= rows_; i++) {
        row_pointers_[i] = arg.row_pointers_[i];
    }

    for (int i = 0; i < nnz_; i++) {
        col_indices_[i] = arg.col_indices_[i];
        values_[i] = arg.values_[i];
    }
}

SparseMatrix::SparseMatrix(int size, double* diagonalValues, const char* s) : rows_(size), cols_(size) {
    if (s != "diag") return;
    // 対角行列の場合、非ゼロ要素の数は size となります
    nnz_ = size;

    // 行ポインタ、列インデックス、データを初期化
    row_pointers_ = new int[size + 1];
    col_indices_ = new int[size];
    values_ = new double[size];

    // 各要素の初期化
    for (int i = 0; i < size; i++) {
        row_pointers_[i] = i;
        col_indices_[i] = i;
        values_[i] = diagonalValues[i];
    }

    // 行ポインタの最後を設定
    row_pointers_[size] = size;
}

// デストラクタ
SparseMatrix::~SparseMatrix() {
    // データを解放
    delete[] row_pointers_;
    delete[] col_indices_;
    delete[] values_;
}

double& SparseMatrix::operator()(int row, int index) { return values_[row_pointers_[row] + index]; }

double SparseMatrix::operator()(int row, int index) const { return values_[row_pointers_[row] + index]; }

double& SparseMatrix::value(int row, int index) { return values_[row_pointers_[row] + index]; }

int& SparseMatrix::dense_index(int row, int index) { return col_indices_[row_pointers_[row] + index]; }

//(i,j,"index") : スパースでi行目，j番目の要素の、スパースではない本来の列番号
int& SparseMatrix::operator()(int row, int index, const char* s) {
    if (strcmp(s, "index") != 0) {
        std::cerr << "Invalid string parameter!" << std::endl;
        exit(1);
    }
    return col_indices_[row_pointers_[row] + index];
}

//(row,s) : row行目の要素数
int SparseMatrix::operator()(int row, const char* s) const {
    if (strcmp(s, "row") != 0) {
        std::cerr << "Invalid string parameter!!" << std::endl;
        exit(1);
    }
    int result = row_pointers_[row + 1] - row_pointers_[row];
    return result;
}

int SparseMatrix::rows() const { return rows_; }
int SparseMatrix::cols() const { return cols_; }
int SparseMatrix::nnz() const { return nnz_; }  // total要素数

int SparseMatrix::nnz(int row) {
    int result = row_pointers_[row + 1] - row_pointers_[row];
    return result;
}

SparseMatrix SparseMatrix::remove_zeros(){
    // 非ゼロ要素の数を数える
    int non_zero_count = 0;
    for (int i = 0; i < nnz_; i++) {
        if (values_[i] != 0.0) {
            non_zero_count++;
        }
    }

    // 新しいスパースマトリクスを作成
    SparseMatrix non_zero_matrix(rows_, cols_, non_zero_count);

    // 新しいスパースマトリクスの行ポインタ、列インデックス、データ
    int* new_row_pointers = new int[rows_ + 1];
    int* new_col_indices = new int[non_zero_count];
    double* new_values = new double[non_zero_count];

    int current_non_zero_index = 0;
    new_row_pointers[0] = 0;

    for (int i = 0; i < rows_; i++) {
        for (int j = row_pointers_[i]; j < row_pointers_[i + 1]; j++) {
            if (values_[j] != 0.0) {
                new_col_indices[current_non_zero_index] = col_indices_[j];
                new_values[current_non_zero_index] = values_[j];
                current_non_zero_index++;
            }
        }
        new_row_pointers[i + 1] = current_non_zero_index;
    }

    // 新しいスパースマトリクスに行ポインタ、列インデックス、データをセット
    non_zero_matrix.set_row_pointers(new_row_pointers);
    non_zero_matrix.set_col_indices(new_col_indices);
    non_zero_matrix.set_values(new_values);
    non_zero_matrix.set_nnz(non_zero_count);

    return non_zero_matrix;
}

SparseMatrix& SparseMatrix::operator=(const SparseMatrix& arg) {
    if (this == &arg) {
        return *this;  // 自己代入の場合、何もしない
    }

    // 新しいリソースを一時的に確保
    int* new_row_pointers = new int[arg.rows_ + 1];
    int* new_col_indices = new int[arg.nnz_];
    double* new_values = new double[arg.nnz_];

    // メンバー変数をコピー
    for (int i = 0; i <= arg.rows_; i++) {
        new_row_pointers[i] = arg.row_pointers_[i];
    }

    for (int i = 0; i < arg.nnz_; i++) {
        new_col_indices[i] = arg.col_indices_[i];
        new_values[i] = arg.values_[i];
    }

    // 既存のリソースを解放
    delete[] row_pointers_;
    delete[] col_indices_;
    delete[] values_;

    // メンバー変数を新しいリソースに設定
    row_pointers_ = new_row_pointers;
    col_indices_ = new_col_indices;
    values_ = new_values;
    rows_ = arg.rows_;
    cols_ = arg.cols_;
    nnz_ = arg.nnz_;

    return *this;
}

// ムーブ代入演算子の実装
SparseMatrix& SparseMatrix::operator=(SparseMatrix&& arg) {
    if (this == &arg) {
        return *this;  // 自己代入の場合、何もしない
    }

    // 既存のリソースを解放
    delete[] row_pointers_;
    delete[] col_indices_;
    delete[] values_;

    // メンバー変数をムーブ
    rows_ = arg.rows_;
    cols_ = arg.cols_;
    nnz_ = arg.nnz_;
    row_pointers_ = arg.row_pointers_;
    col_indices_ = arg.col_indices_;
    values_ = arg.values_;

    // 右辺値のリソースを無効化
    arg.rows_ = 0;
    arg.cols_ = 0;
    arg.nnz_ = 0;
    arg.row_pointers_ = nullptr;
    arg.col_indices_ = nullptr;
    arg.values_ = nullptr;

    return *this;
}

Matrix SparseMatrix::operator*(Matrix& arg) {
    int numRowsResult = rows_;
    int numColsResult = arg.cols();
    Matrix result(numRowsResult, numColsResult);
    double* dataA = (*this).get_values();
    int* dataArowpointers = (*this).get_row_pointers();
    int* dataAcolindices = (*this).get_col_indices();
    double* dataB = arg.get_values();
    double* dataResult = result.get_values();

    for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < arg.cols(); j++) {
            result(i, j) = 0.0;
        }
        for (int k = 0; k < (*this)(i, "row"); k++) {
            for (int j = 0; j < arg.cols(); j++) {
                result(i, j) += (*this)(i, k) * arg[(*this)(i, k, "index")][j];
            }
        }
    }

    return result;
}

void SparseMatrix::print_values() {
    int rows = (*this).rows();
    int cols = (*this).cols();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < *((*this).get_row_pointers() + i + 1) - *((*this).get_row_pointers() + i); ++j) {
            std::cout << "(" << i << ", " << *((*this).get_col_indices() + *((*this).get_row_pointers() + i) + j) << ") : " << (*this)(i, j) << "\t";
        }
        std::cout << std::endl;
    }
}

double* SparseMatrix::get_values() { return values_; }
int* SparseMatrix::get_row_pointers() { return row_pointers_; }
int* SparseMatrix::get_col_indices() { return col_indices_; }

void SparseMatrix::set_row_pointers(int* new_row_pointers) {
    // 以前のメモリを解放
    if (row_pointers_ != nullptr) delete[] row_pointers_;

    // 新しいポインタを設定
    row_pointers_ = new_row_pointers;
}

// col_indicesをセットするメンバ関数
void SparseMatrix::set_col_indices(int* new_col_indices) {
    // 以前のメモリを解放
    if (col_indices_ != nullptr) delete[] col_indices_;

    // 新しいポインタを設定
    col_indices_ = new_col_indices;
}

// dataをセットするメンバ関数
void SparseMatrix::set_values(double* new_values) {
    // 以前のメモリを解放
    if (values_ != nullptr) delete[] values_;

    // 新しいポインタを設定
    values_ = new_values;
}

// dataをセットするメンバ関数
void SparseMatrix::set_nnz(int nnz) { nnz_ = nnz; }

SparseMatrix SparseMatrix::transpose() const {
    // 転置行列の行数と列数は元の行列の列数と行数になります
    SparseMatrix transposed(cols_, rows_);

    // 転置行列の行ポインタ
    int* transposed_row_pointers = new int[cols_ + 1];

    // 各列に含まれる非ゼロ要素の数を数える
    int* count_non_zeros = new int[cols_];
    for (int i = 0; i < cols_; i++) {
        count_non_zeros[i] = 0;
    }
    for (int i = 0; i < nnz_; i++) {
        count_non_zeros[col_indices_[i]]++;
    }

    // 転置行列の行ポインタを設定
    transposed_row_pointers[0] = 0;
    for (int i = 1; i <= cols_; i++) {
        transposed_row_pointers[i] = transposed_row_pointers[i - 1] + count_non_zeros[i - 1];
    }

    // 転置行列の列インデックスとデータ
    int* transposed_col_indices = new int[nnz_];
    double* transposed_values = new double[nnz_];

    // 転置行列を生成
    for (int i = 0; i < rows_; i++) {
        for (int j = row_pointers_[i]; j < row_pointers_[i + 1]; j++) {
            int col = col_indices_[j];
            int dest = transposed_row_pointers[col];

            transposed_col_indices[dest] = i;
            transposed_values[dest] = values_[j];

            transposed_row_pointers[col]++;
        }
    }

    // 転置行列に行ポインタ、列インデックス、データをセット
    transposed.set_row_pointers(transposed_row_pointers);
    transposed.set_col_indices(transposed_col_indices);
    transposed.set_values(transposed_values);
    transposed.set_nnz(nnz_);

    // 元の行列から作成したポインタを解放
    delete[] count_non_zeros;

    return transposed;
}
