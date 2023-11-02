#include "sparseVector.h"

#include "vector.h"

// コンストラクタ
SparseVector::SparseVector(int size, int nnz) try : size_(size), nnz_(nnz), indices_(new int[nnz]), values_(new double[nnz]) {
} catch (std::bad_alloc) {
    std::cerr << "SparseVector::SparseVector(int nnz_): Out of Memory!" << std::endl;
    throw;
}

// コピーコンストラクタ
SparseVector::SparseVector(const SparseVector &arg) try
    : size_(arg.size_), nnz_(arg.nnz_), indices_(new int[nnz_]), values_(new double[nnz_]) {
    for (int i = 0; i < nnz_; i++) {
        indices_[i] = arg.indices_[i];
        values_[i] = arg.values_[i];
    }
} catch (std::bad_alloc) {
    std::cerr << "SparseVector::SparseVector(const SparseVector &): Out of Memory!" << std::endl;
    throw;
}

// ムーブコンストラクタ
SparseVector::SparseVector(SparseVector &&arg) : size_(arg.size_), nnz_(arg.nnz_), indices_(arg.indices_), values_(arg.values_) {
    arg.size_ = 0;
    arg.nnz_ = 0;
    arg.indices_ = nullptr;
    arg.values_ = nullptr;
}

// デストラクタ
SparseVector::~SparseVector(void) {
    delete[] indices_;
    delete[] values_;
}

// コピー代入
SparseVector &SparseVector::operator=(const SparseVector &arg) {
    if (this == &arg) return *this;
    if (this->nnz_ != arg.nnz_) {
        nnz_ = arg.nnz_;
        delete[] indices_;
        delete[] values_;
        try {
            indices_ = new int[nnz_];
            values_ = new double[nnz_];
        } catch (std::bad_alloc) {
            std::cerr << "Out of Memory" << std::endl;
            throw;
        }
    }
    size_ = arg.size_;
    for (int i = 0; i < nnz_; i++) {
        indices_[i] = arg.indices_[i];
        values_[i] = arg.values_[i];
    }
    return *this;
}

// ムーブ代入
SparseVector &SparseVector::operator=(SparseVector &&arg) {
    if (this == &arg) return *this;
    size_ = arg.size_;
    nnz_ = arg.nnz_;
    delete[] indices_;
    delete[] values_;
    values_ = arg.values_;
    indices_ = arg.indices_;
    arg.nnz_ = 0;
    arg.indices_ = nullptr;
    arg.values_ = nullptr;
    return *this;
}

int SparseVector::size(void) const { return size_; }

int SparseVector::nnz(void) const { return nnz_; }

int SparseVector::indices_indices_(int indices_) const { return indices_[indices_]; }

int &SparseVector::indices_indices_(int indices_) { return indices_[indices_]; }

double SparseVector::values_indices_(int indices_) const { return values_[indices_]; }

double &SparseVector::values_indices_(int indices_) { return values_[indices_]; }

SparseVector SparseVector::operator+(void) const { return *this; }

SparseVector SparseVector::operator-(void) const {
    SparseVector result = *this;
    for (int i = 0; i < result.nnz_; i++) result.values_indices_(i) *= -1.0;
    return result;
}

bool SparseVector::operator==(const SparseVector &rhs) const {
    if (Size != rhs.Size || nnz_ != rhs.nnz_()) return false;
    for (int i = 0; i < nnz_; i++) {
        if (values_[i] != rhs.values_indices_(i) || indices_[i] != rhs.indices_indices_(i)) return false;
    }
    return true;
}

bool SparseVector::operator!=(const SparseVector &rhs) const {
    if (Size != rhs.Size || nnz_ != rhs.nnz_()) return true;
    for (int i = 0; i < nnz_; i++) {
        if (values_[i] != rhs.values_indices_(i) || indices_[i] != rhs.indices_indices_(i)) return true;
    }
    return false;
}

void SparseVector::modifyvalues_(int n, int indices_, double value) {
    this->values_indices_(n) = value;
    this->indices_indices_(n) = indices_;
    return;
}

std::ostream &operator<<(std::ostream &os, const SparseVector &rhs) {
    os << "(";
    if (rhs.nnz_() > 0) {
        for (int i = 0;; i++) {
            os << rhs.indices_indices_(i) << ":" << rhs.values_indices_(i);
            if (i >= rhs.nnz_() - 1) break;
            os << ", ";
        }
    }
    os << ')';
    return os;
}

double max_norm(const SparseVector &arg) {
    if (arg.nnz_() < 1) {
        std::cout << "Can't calculate norm for 0-sized vector" << std::endl;
        exit(1);
    }
    double result = fabs(arg.values_indices_(0));
    for (int i = 1; i < arg.nnz_(); i++) {
        double tmp = fabs(arg.values_indices_(i));
        if (result < tmp) result = tmp;
    }
    return result;
}

double squared_norm(const SparseVector &arg) { return sqrt(norm_square(arg)); }

double norm_square(const SparseVector &arg) {
    double result = 0.0;
    for (int i = 0; i < arg.nnz_(); i++) {
        result += arg.values_indices_(i) * arg.values_indices_(i);
    }
    return result;
}

double L1norm_square(const SparseVector &arg) {
    double result = 0.0;
    for (int i = 0; i < arg.nnz_(); i++) {
        result += fabs(arg.values_indices_(i));
    }
    return result;
}

double operator*(const SparseVector &lhs, const Vector &rhs) {
    double result = 0.0;
    for (int ell = 0; ell < lhs.nnz_(); ell++) {
        result += lhs.values_indices_(ell) * rhs[lhs.indices_indices_(ell)];
    }
    return result;
}

double operator*(const Vector &lhs, const SparseVector &rhs) {
    double result = 0.0;
    for (int ell = 0; ell < rhs.nnz_(); ell++) {
        result += rhs.values_indices_(ell) * lhs[rhs.indices_indices_(ell)];
    }
    return result;
}
