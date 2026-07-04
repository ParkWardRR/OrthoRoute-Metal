#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    // bindings to Apple's Accelerate framework
    // void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
    //                  const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
    //                  const int K, const float alpha, const float *A, const int lda,
    //                  const float *B, const int ldb, const float beta,
    //                  float *C, const int ldc);
    pub fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}
pub const CBLAS_ROW_MAJOR: i32 = 101;
pub const CBLAS_NO_TRANS: i32 = 111;

/// Multiply two dense matrices using Apple's AMX coprocessor via Accelerate framework.
/// Computes C = alpha * A * B + beta * C
pub fn amx_sgemm(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32], beta: f32, c: &mut [f32]) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            beta,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}
