#!/usr/bin/env python3
import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Compile the CUDA kernel (copy your original cu_source string here)
cu_source = r'''
#include <stdint.h>

struct LUT {
    int a0, b0, a1, b1, cin;
    int a0b0, a1b1;
    int sum, cout;
    bool bit_valid;
    int result;
};

__device__ __forceinline__ void and_ed(LUT* lut) {
    lut->a0b0 = lut->a0 & lut->b0;
    lut->a1b1 = lut->a1 & lut->b1;
}

__device__ __forceinline__ void and_ed_comp_1(LUT* lut) {
    lut->a0b0 = 1 ^ (lut->a0 & lut->b0);
    lut->a1b1 = lut->a1 & lut->b1;
}

__device__ __forceinline__ void and_ed_comp_2(LUT* lut) {
    lut->a0b0 = lut->a0 & lut->b0;
    lut->a1b1 = 1 ^ (lut->a1 & lut->b1);
}

__device__ __forceinline__ void and_ed_comp_3(LUT* lut) {
    lut->a0b0 = lut->a0 & lut->b0;
    lut->a1b1 = 1 ^ (lut->a1 & lut->b1);
}

__device__ __forceinline__ void and_ed_comp_4(LUT* lut) {
    lut->a0b0 = 1 ^ (lut->a0 & lut->b0);
    lut->a1b1 = 1 ^ (lut->a1 & lut->b1);
}

__device__ __forceinline__ void and_ed_comp_5(LUT* lut) {
    lut->a0b0 = lut->a0 & lut->b0;
    lut->a1b1 = lut->a1 & lut->b1;
}

__device__ __forceinline__ void addition(LUT* lut) {
    if (lut->bit_valid) {
        lut->result = lut->a0b0 + lut->a1b1 + lut->cin;
        lut->sum = lut->result % 2;
        lut->cout = lut->result / 2;
    } else {
        lut->sum = 0;
        lut->cout = lut->cin;
    }
}

__device__ __forceinline__ void fxp_to_bin_device(int value, int* bits, int bit_width) {
    unsigned int uval = static_cast<unsigned int>(value);
    for (int i = 0; i < bit_width; ++i) {
        bits[i] = (uval >> (bit_width - 1 - i)) & 1;
    }
}


__device__ __forceinline__ int mult_kernel(int a, int b, uint64_t lut_config) {

    int op1_bin[8], op2_bin[8];
    fxp_to_bin_device(a, op1_bin, 8);
    fxp_to_bin_device(b, op2_bin, 8);

    for (int i = 0; i < 4; ++i) {
        int temp1 = op1_bin[i];
        int temp2 = op2_bin[i];

        op1_bin[i] = op1_bin[7 - i];
        op2_bin[i] = op2_bin[7 - i];

        op1_bin[7 - i] = temp1;
        op2_bin[7 - i] = temp2;
    }

    //reverse_array(op1_bin, 8);
    //reverse_array(op2_bin, 8);

    LUT lut_list[4][9];
    int pps_list[4][17] = {0};

    int list_ind = 0;

    for (int mlr = 0; mlr < 6; mlr += 2) {
        lut_list[list_ind][0].a0 = op1_bin[0];
        lut_list[list_ind][0].b0 = op2_bin[mlr];
        lut_list[list_ind][0].a1 = 0;
        lut_list[list_ind][0].b1 = 0;
        lut_list[list_ind][0].cin = 0;
        int pos = list_ind * 9 + 0;
        lut_list[list_ind][0].bit_valid = (lut_config >> (35 - pos)) & 1;
        and_ed(&lut_list[list_ind][0]);
        addition(&lut_list[list_ind][0]);

        for (int mtd = 1; mtd < 7; ++mtd) {
            lut_list[list_ind][mtd].a0 = op1_bin[mtd];
            lut_list[list_ind][mtd].b0 = op2_bin[mlr];
            lut_list[list_ind][mtd].a1 = op1_bin[mtd-1];
            lut_list[list_ind][mtd].b1 = op2_bin[mlr+1];
            lut_list[list_ind][mtd].cin = lut_list[list_ind][mtd-1].cout;
            pos = list_ind * 9 + mtd;
            lut_list[list_ind][mtd].bit_valid = (lut_config >> (35 - pos)) & 1;
            and_ed(&lut_list[list_ind][mtd]);
            addition(&lut_list[list_ind][mtd]);
        }

        int mtd = 7;
        lut_list[list_ind][mtd].a0 = op1_bin[mtd];
        lut_list[list_ind][mtd].b0 = op2_bin[mlr];
        lut_list[list_ind][mtd].a1 = op1_bin[mtd-1];
        lut_list[list_ind][mtd].b1 = op2_bin[mlr+1];
        lut_list[list_ind][mtd].cin = lut_list[list_ind][mtd-1].cout;
        pos = list_ind * 9 + mtd;
        lut_list[list_ind][mtd].bit_valid = (lut_config >> (35 - pos)) & 1;
        and_ed_comp_1(&lut_list[list_ind][mtd]);
        addition(&lut_list[list_ind][mtd]);

        mtd = 8;
        lut_list[list_ind][mtd].a0 = 0;
        lut_list[list_ind][mtd].b0 = 0;
        lut_list[list_ind][mtd].a1 = op1_bin[7];
        lut_list[list_ind][mtd].b1 = op2_bin[mlr+1];
        lut_list[list_ind][mtd].cin = lut_list[list_ind][7].cout;
        pos = list_ind * 9 + mtd;
        lut_list[list_ind][mtd].bit_valid = (lut_config >> (35 - pos)) & 1;
        and_ed_comp_2(&lut_list[list_ind][mtd]);
        addition(&lut_list[list_ind][mtd]);

        for (int j = 0; j <= mtd; ++j) {
            pps_list[list_ind][j] = lut_list[list_ind][j].sum;
        }
        pps_list[list_ind][mtd+1] = lut_list[list_ind][mtd].cout;

        list_ind++;
    }

    // Last Partial Product
    int mlr = 6;
    lut_list[list_ind][0].a0 = op1_bin[0];
    lut_list[list_ind][0].b0 = op2_bin[mlr];
    lut_list[list_ind][0].a1 = 0;
    lut_list[list_ind][0].b1 = 0;
    lut_list[list_ind][0].cin = 0;
    int pos = list_ind * 9 + 0;
    lut_list[list_ind][0].bit_valid = (lut_config >> (35 - pos)) & 1;
    and_ed(&lut_list[list_ind][0]);
    addition(&lut_list[list_ind][0]);

    for (int mtd = 1; mtd < 7; ++mtd) {
        lut_list[list_ind][mtd].a0 = op1_bin[mtd];
        lut_list[list_ind][mtd].b0 = op2_bin[mlr];
        lut_list[list_ind][mtd].a1 = op1_bin[mtd-1];
        lut_list[list_ind][mtd].b1 = op2_bin[mlr+1];
        lut_list[list_ind][mtd].cin = lut_list[list_ind][mtd-1].cout;
        pos = list_ind * 9 + mtd;
        lut_list[list_ind][mtd].bit_valid = (lut_config >> (35 - pos)) & 1;
        and_ed_comp_3(&lut_list[list_ind][mtd]);
        addition(&lut_list[list_ind][mtd]);
    }

    int mtd = 7;
    lut_list[list_ind][mtd].a0 = op1_bin[mtd];
    lut_list[list_ind][mtd].b0 = op2_bin[mlr];
    lut_list[list_ind][mtd].a1 = op1_bin[mtd-1];
    lut_list[list_ind][mtd].b1 = op2_bin[mlr+1];
    lut_list[list_ind][mtd].cin = lut_list[list_ind][mtd-1].cout;
    pos = list_ind * 9 + mtd;
    lut_list[list_ind][mtd].bit_valid = (lut_config >> (35 - pos)) & 1;
    and_ed_comp_4(&lut_list[list_ind][mtd]);
    addition(&lut_list[list_ind][mtd]);

    mtd = 8;
    lut_list[list_ind][mtd].a0 = 0;
    lut_list[list_ind][mtd].b0 = 0;
    lut_list[list_ind][mtd].a1 = op1_bin[7];
    lut_list[list_ind][mtd].b1 = op2_bin[mlr+1];
    lut_list[list_ind][mtd].cin = lut_list[list_ind][7].cout;
    pos = list_ind * 9 + mtd;
    lut_list[list_ind][mtd].bit_valid = (lut_config >> (35 - pos)) & 1;
    and_ed_comp_5(&lut_list[list_ind][mtd]);
    addition(&lut_list[list_ind][mtd]);

    for (int j = 0; j <= mtd; ++j) {
        pps_list[list_ind][j] = lut_list[list_ind][j].sum;
    }
    pps_list[list_ind][mtd+1] = lut_list[list_ind][mtd].cout;

    int final_product = 0;
    for (int i = 0; i < 4; ++i) {
        int pp_value = 0;
//        int pp_length = 0;
        for (int j = 0; j < 17; ++j) {
            if (pps_list[i][j]) {
                pp_value += pps_list[i][j] << j;
            }
        }
        final_product += pp_value << (2 * i);
    }

    final_product += (1 << 8) + (1 << 15);
    final_product -= (1 << 16);
    //printf("a_val = %d  b_val = %d final_product = %d accurate = %d\n", a, b, final_product, a*b);
    //if (final_product != a*b)
    //    printf(" ERROR = \n");


    return final_product;
}
extern "C"
__global__ void compute_all(const int8_t* d_a, const int8_t* d_b,
                            uint64_t lut_config, int32_t* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d_out[idx] = mult_kernel(d_a[idx], d_b[idx], lut_config);
}
'''

mod = SourceModule(cu_source, options=["-std=c++14"], no_extern_c=True)
compute_all = mod.get_function("compute_all")

# Prepare input data
bit_width = 8
n = (1 << bit_width) * (1 << bit_width)
vals = np.arange(-(1 << (bit_width - 1)), (1 << (bit_width - 1)), dtype=np.int8)
aa, bb = np.meshgrid(vals, vals, indexing="ij")
h_a = aa.ravel()
h_b = bb.ravel()

d_a = cuda.mem_alloc(h_a.nbytes)
d_b = cuda.mem_alloc(h_b.nbytes)
cuda.memcpy_htod(d_a, h_a)
cuda.memcpy_htod(d_b, h_b)

h_out = np.empty(n, dtype=np.int32)
d_out = cuda.mem_alloc(h_out.nbytes)

def run_kernel(lut_config):
    threads = 256
    blocks = (n + threads - 1) // threads

    compute_all(
        d_a, d_b,
        np.uint64(lut_config),
        d_out,
        np.int32(n),
        block=(threads, 1, 1),
        grid=(blocks, 1)
    )
    cuda.Context.synchronize()
    cuda.memcpy_dtoh(h_out, d_out)

    return h_out.copy()

def Compute_Metrics(lut_val):
    results = run_kernel(lut_val)
    actuals = h_a.astype(np.int64) * h_b.astype(np.int64)
    errors = results.astype(np.int64) - actuals
    abs_err = np.abs(errors)

    rel = np.zeros_like(errors, dtype=np.float64)
    mask = actuals != 0
    rel[mask] = errors[mask] / actuals[mask]
    abs_rel = np.abs(rel)

    metrics = {
        "avg_error": errors.mean(),
        "avg_abs_error": abs_err.mean(),
        "avg_rel_error": rel.mean(),
        "avg_abs_rel_error": abs_rel.mean(),
        "max_error": errors.max(),
        "min_error": errors.min(),
        "error_probability": np.count_nonzero(errors) / errors.size,
    }
    return metrics

# if __name__ == "__main__":

#     lut_val = 404750335
    

#     print(f"Running kernel with LUT config: {hex(lut_val)} ({lut_val})")
#     output = run_kernel(lut_val)
#     metrics = Compute_Metrics(lut_val)

#     print("\n--- Error Metrics ---")
#     for k, v in metrics.items():
#         print(f"{k}: {v}")
