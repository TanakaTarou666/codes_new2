#include "sparse_matrix.h"
#include "sparse_vector.h"

#ifndef __DSDTENSOR__
#define __DSDTENSOR__

class DSSTensor {
   private:
    int rows_;
    int cols_;
    int depth_;
    int nnz_;
    int* row_pointers_;
    int* col_indices_;
    SparseVector* elements_;

    public:
    DSSTensor(SparseMatrix &arg,int depth);
    DSSTensor(SparseMatrix &arg,int depth,SparseVector* elements);
    DSSTensor();
    ~DSSTensor();
    SparseVector& operator()(int row, int col); 
    SparseVector operator()(int row, int col) const;  
    int& operator()(int row, int index, const char* s);
    int operator()(int row, int index, const char* s) const;
    int operator()(int row, const char* s) const;
    int& dense_index(int row,int index);
    DSSTensor& operator=(const DSSTensor& arg);  // コピー代入演算子
    DSSTensor& operator=(DSSTensor&& arg); 
    int rows() const;
    int cols() const;
    int depth() const;
    int nnz() const;
    int nnz(int row) const;
    SparseVector* get_elements();  // データへのポインタを取得するメソッド
    int* get_row_pointers();  // データへのポインタを取得するメソッド
    int* get_col_indices();  // データへのポインタを取得するメソッド
};

#endif