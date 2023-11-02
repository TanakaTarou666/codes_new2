
#ifndef __VECTOR__
#define __VECTOR__

class Vector {
   private:
    double* values_;       // Change: Renamed from Element
    int size_;             // Change: Renamed from Size
    bool part_of_matrix_;  // New member

   public:
    Vector(void);
    explicit Vector(int n);
    Vector(double* data, int size);
    Vector(int n, double value, const char* flag);
    Vector(const Vector& arg);
    Vector& operator=(const Vector& rhs);
    ~Vector(void);
    int size(void) const;
    double operator[](int index) const;
    double& operator[](int index);
    std::ostream& print(std::ostream& lhs) const;
    Vector& operator+=(const Vector& rhs);
    Vector& operator-=(const Vector& rhs);
    Vector operator+(void) const;
    Vector operator-(void) const;
    double* get_values();  // データへのポインタを取得するメソッド
};

std::ostream& operator<<(std::ostream& lhs, const Vector& rhs);
Vector operator+(const Vector& lhs, const Vector& rhs);
Vector operator-(const Vector& lhs, const Vector& rhs);
double operator*(const Vector& lhs, const Vector& rhs);
Vector operator*(double lhs, const Vector& rhs);
Vector operator/(const Vector& lhs, double rhs);
bool operator==(const Vector& lhs, const Vector& rhs);
bool operator!=(const Vector& lhs, const Vector& rhs);
double norm(const Vector& arg, int p);
double max_norm(const Vector& arg);
double squared_norm(const Vector& arg);

#endif