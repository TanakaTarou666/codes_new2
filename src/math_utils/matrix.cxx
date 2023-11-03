#include "matrix.h"

Matrix::Matrix(int rows, int cols)
    : rows_(rows), cols_(cols), values_(new double[rows * cols]) {}

Matrix::Matrix(int rows, int cols, double arg)
    : rows_(rows), cols_(cols), values_(new double[rows * cols]) {
    for (int i = 0; i < rows * cols; i++) {
        values_[i] = arg;
    }
}

Matrix::Matrix(void) : rows_(0), cols_(0), values_(NULL) {}
// Copy constructor
Matrix::Matrix(const Matrix& other)
    : rows_(other.rows_),
      cols_(other.cols_),
      values_(new double[other.rows_ * other.cols_]) {
    for (int i = 0; i < rows_ * cols_; i++) {
        values_[i] = other.values_[i];
    }
}

// Destructor
Matrix::~Matrix() { delete[] values_; }

// Assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            delete[] values_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            values_ = new double[rows_ * cols_];
        }
        for (int i = 0; i < rows_ * cols_; i++) {
            values_[i] = other.values_[i];
        }
    }
    return *this;
}

// Getters for rows and columns
int Matrix::rows() const { return rows_; }

int Matrix::cols() const { return cols_; }

// Element access operators
double& Matrix::operator()(int row, int col) {
    return values_[row * cols_ + col];
}

double Matrix::operator()(int row, int col) const {
    return values_[row * cols_ + col];
}

Vector Matrix::operator[](int row){
    if (row >= 0 && row < rows_) {
        return Vector(values_ + row * cols_, cols_);
    } else {
        throw std::out_of_range("Row index out of range");
    }
}

// Unary operators
Matrix Matrix::operator+() const { return *this; }

Matrix Matrix::operator-() const {
    Matrix result(*this);
    for (int i = 0; i < rows_ * cols_; i++) {
        result.values_[i] *= -1.0;
    }
    return result;
}

// Compound assignment operators
Matrix& Matrix::operator+=(const Matrix& rhs) {
    if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
        std::cerr << "Matrix::operator+=: Size Unmatched" << std::endl;
        exit(1);
    }
    for (int i = 0; i < rows_ * cols_; i++) {
        values_[i] += rhs.values_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
    if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
        std::cerr << "Matrix::operator-=: Size Unmatched" << std::endl;
        exit(1);
    }
    for (int i = 0; i < rows_ * cols_; i++) {
        values_[i] -= rhs.values_[i];
    }
    return *this;
}

std::ostream& Matrix::print(std::ostream& lhs) const {
    lhs << "(";
    int i;
    for (int row = 0; row < rows_ - 1; row++) {
        lhs << "(";
        for (int col = 0; col < cols_; col++) {
            lhs << values_[row * cols_ + col] << ", ";
        }
        lhs << ")" << std::endl;
    }
    lhs << "(";
    for (int col = 0; col < cols_; col++) {
        lhs << values_[(rows_ - 1) * cols_ + col] << ", ";
    }
    lhs << "))";
    return lhs;
}

double* Matrix::get_values() { return values_; }

std::ostream& operator<<(std::ostream& lhs, const Matrix& rhs) {
    return rhs.print(lhs);
}

Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs);
    return (result += rhs);
}

Matrix operator-(const Matrix& lhs, const Matrix& rhs) {
    Matrix result(lhs);
    return (result -= rhs);
}

Matrix operator*(double factor, const Matrix& rhs) {
    if (rhs.rows() == 0 || rhs.cols() == 0) {
        std::cerr << "operator*(double , const Matrix &): Size unmatched"
                  << std::endl;
        exit(1);
    }
    Matrix result(rhs);
    for (int row = 0; row < result.rows(); row++) {
        for (int col = 0; col < result.cols(); col++) {
            result(row, col) *= factor;
        }
    }
    return result;
}

Vector operator*(const Matrix& lhs, const Vector& rhs) {
    if (lhs.cols() != rhs.size() || lhs.rows() == 0) {
        std::cerr << "operator*(const Matrix &, const Vector &): Size unmatched"
                  << std::endl;
        exit(1);
    }
    Vector result(lhs.rows());
    for (int row = 0; row < lhs.rows(); row++) {
        result[row] = 0.0;
        for (int col = 0; col < lhs.cols(); col++) {
            result[row] += lhs(row, col) * rhs[col];
        }
    }
    return result;
}

// アドレスに直接アクセスすることで処理速度を上げている
Matrix operator*(Matrix& lhs, Matrix& rhs) {
    if (lhs.cols() != rhs.rows() || lhs.rows() == 0 || rhs.cols() == 0) {
        std::cerr << "operator*(const Matrix &, const Matrix &): Size unmatched"
                  << std::endl;
        exit(1);
    }

    int numrows_A = lhs.rows();
    int numcols_A = lhs.cols();
    int numrows_B = rhs.rows();
    int numcols_B = rhs.cols();

    Matrix result(lhs.rows(), rhs.cols());

    int numrows_Result = numrows_A;
    int numcols_Result = numcols_B;

    double* values_A = lhs.get_values();
    double* values_B = rhs.get_values();
    double* values_Result = result.get_values();

    for (int i = 0; i < numrows_Result; ++i) {
        for (int j = 0; j < numcols_Result; ++j) {
            *values_Result++ = 0.0;
        }
        for (int k = 0; k < numcols_A; ++k) {
            values_Result -= numcols_Result;
            for (int j = 0; j < numcols_Result; ++j) {
                *values_Result++ += *values_A * *values_B++;
            }
            values_A++;
        }
        values_B -= numcols_A * numcols_Result;
    }
    return result;
}

/*
Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.cols() != rhs.rows() || lhs.rows() == 0 || rhs.cols() == 0) {
        std::cerr << "operator*(const Matrix &, const Matrix &): Size unmatched"
<< std::endl; exit(1);
    }
    Matrix result(lhs.rows(), rhs.cols());
    for (int row = 0; row < result.rows(); row++) {
        for (int col = 0; col < result.cols(); col++) {
            result(row, col) = 0.0;
        }
    }
    for (int row = 0; row < result.rows(); row++) {
        for (int col = 0; col < result.cols(); col++) {
            for (int k = 0; k < lhs.cols(); k++) {
                result(row, col) += lhs(row, k) * rhs(k, col);
            }
        }
    }
    return result;
}
*/

bool operator==(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        return false;
    }
    for (int row = 0; row < lhs.rows(); row++) {
        for (int col = 0; col < lhs.cols(); col++) {
            if (lhs(row, col) != rhs(row, col)) {
                return false;
            }
        }
    }
    return true;
}

bool operator!=(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        return true;
    }
    for (int row = 0; row < lhs.rows(); row++) {
        for (int col = 0; col < lhs.cols(); col++) {
            if (lhs(row, col) != rhs(row, col)) {
                return true;
            }
        }
    }
    return false;
}

double frobenius_norm(const Matrix& arg) {
    double result = 0.0;
    for (int row = 0; row < arg.rows(); row++) {
        for (int col = 0; col < arg.cols(); col++) {
            result += arg(row, col) * arg(row, col);
        }
    }
    return sqrt(result);
}

Matrix transpose(const Matrix& arg) {
    if (arg.rows() == 0 || arg.cols() == 0) {
        std::cerr << "transpose(const Matrix): zero-sized matrix" << std::endl;
    }
    Matrix result(arg.cols(), arg.rows());
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.cols(); j++) {
            result(i, j) = arg(j, i);
        }
    }
    return result;
}
