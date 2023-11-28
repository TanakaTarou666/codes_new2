#include "vector.h"

Vector::Vector(void) : size_(0), values_(nullptr), part_of_matrix_(false) {}

Vector::Vector(double* values, int size) : values_(values), size_(size) { part_of_matrix_ = true; };

Vector::Vector(int n) try : size_(n), values_(new double[n]), part_of_matrix_(false) {
} catch (std::bad_alloc) {
    std::cerr << "Vector::Vector(int n) : Out of Memory" << std::endl;
    std::cerr << "n:" << n << std::endl;
    throw;
}

Vector::Vector(int n, double value, const char* flag) try : size_(n), values_(new double[n]), part_of_matrix_(false) {
    if (strcmp(flag, "all") != 0) {
        std::cerr << "Unknown option: \"" << flag << "\"" << std::endl;
        throw;
    }
    for (int i = 0; i < size_; i++) {
        values_[i] = value;
    }
} catch (std::bad_alloc) {
    std::cerr << "Vector::Vector(int n, double value, const char* flag) : Out of Memory" << std::endl;
    std::cerr << "n:" << n << std::endl;
    throw;
}

Vector::Vector(const Vector& arg) try : size_(arg.size()), values_(new double[arg.size()]), part_of_matrix_(false) {
    for (int i = 0; i < size_; i++) {
        values_[i] = arg.values_[i];
    }
} catch (std::bad_alloc) {
    std::cerr << "Vector::Vector(const Vector& arg) : Out of Memory" << std::endl;
    throw;
}

Vector& Vector::operator=(const Vector& rhs) {
    if (this != &rhs) {
        if (size_ != rhs.size()) {
            size_ = rhs.size();
            delete[] values_;
            try {
                values_ = new double[size_];
            } catch (std::bad_alloc) {
                std::cerr << "Vector::operator=: Out of Memory" << std::endl;
                throw;
            }
        }
        for (int i = 0; i < size_; i++) {
            values_[i] = rhs.values_[i];
        }
    }
    return *this;
}

Vector::~Vector(void) {
    if (part_of_matrix_ == false) delete[] values_;
}

int Vector::size(void) const { return size_; }

double Vector::operator[](int index) const { return values_[index]; }

double& Vector::operator[](int index) { return values_[index]; }

Vector& Vector::operator+=(const Vector& rhs) {
    if (size_ != rhs.size()) {
        std::cerr << "Vector::operator+=: Size Unmatched" << std::endl;
        exit(1);
    }
    for (int i = 0; i < size_; i++) {
        values_[i] += rhs[i];
    }
    return *this;
}

Vector& Vector::operator-=(const Vector& rhs) {
    if (size_ != rhs.size()) {
        std::cerr << "Vector::operator-=: Size Unmatched" << std::endl;
        exit(1);
    }
    for (int i = 0; i < size_; i++) {
        values_[i] -= rhs[i];
    }
    return *this;
}

std::ostream& Vector::print(std::ostream& lhs) const {
    lhs << "(";
    for (int i = 0; i < size_; i++) {
        lhs << values_[i] << ", ";
    }
    lhs << ")";
    return lhs;
}

Vector Vector::operator+(void) const { return *this; }

Vector Vector::operator-(void) const {
    Vector result(*this);
    for (int i = 0; i < size_; i++) {
        result[i] = -result[i];
    }
    return result;
}

std::ostream& operator<<(std::ostream& lhs, const Vector& rhs) { return rhs.print(lhs); }

Vector operator+(const Vector& lhs, const Vector& rhs) {
    Vector result(lhs);
    return result += rhs;
}

Vector operator-(const Vector& lhs, const Vector& rhs) {
    Vector result(lhs);
    return result -= rhs;
}

double* Vector::get_values() { return values_; }

double operator*(const Vector& lhs, const Vector& rhs) {
    if (lhs.size() != rhs.size()) {
        std::cerr << "operator*(const Vector &, const Vector &): Size Unmatched" << std::endl;
        exit(1);
    }
    double result = 0.0;
    for (int i = 0; i < lhs.size(); i++) {
        result += lhs[i] * rhs[i];
    }
    return result;
}

Vector operator*(double lhs, const Vector& rhs) {
    Vector result(rhs);
    for (int i = 0; i < result.size(); i++) {
        result[i] = lhs * rhs[i];
    }
    return result;
}

Vector operator/(const Vector& lhs, double rhs) {
    Vector result(lhs);
    for (int i = 0; i < result.size(); i++) {
        result[i] = lhs[i] / rhs;
    }
    return result;
}

bool operator==(const Vector& lhs, const Vector& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (int i = 0; i < lhs.size(); i++) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

bool operator!=(const Vector& lhs, const Vector& rhs) {
    if (lhs.size() != rhs.size()) {
        return true;
    }
    for (int i = 0; i < lhs.size(); i++) {
        if (lhs[i] != rhs[i]) {
            return true;
        }
    }
    return false;
}

double squared_sum(const Vector& arg) {
    double result = 0.0;
    for (int i = 0; i < arg.size(); i++) {
        result += arg[i] * arg[i];
    }
    return result;
}

double norm(const Vector& arg, int p) {
    double result = 0.0;
    for (int i = 0; i < arg.size(); i++) {
        result += pow(fabs(arg[i]), p);
    }
    result = pow(result, 1.0 / (double)p);
    return result;
}

double max_norm(const Vector& arg) {
    double result = fabs(arg[0]);
    for (int i = 1; i < arg.size(); i++) {
        double tmp = fabs(arg[i]);
        if (result < tmp) {
            result = tmp;
        }
    }
    return result;
}

double squared_norm(const Vector& arg) {
    return sqrt(squared_sum(arg));
}
