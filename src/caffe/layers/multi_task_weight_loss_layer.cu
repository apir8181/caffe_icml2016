#include <algorithm>
#include <cfloat>
#include <vector>
#include <climits>

#include "caffe/layers/multi_task_weight_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CalculateKernelGPU(const int nthreads,
          const Dtype* data, Dtype* square_diff, int num, int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      int i = index / dim / num;
      int j = index / dim % num;
      int k = index % dim;
      
      Dtype diff = data[i * dim + k] - data[j * dim + k];
      square_diff[(i * num + j) * dim + k] = diff * diff;
  }
}

template <typename Dtype>
__global__ void CalculateLossGPU(const int nthreads,
          const Dtype* kernel, const Dtype* task_index, 
          const Dtype* Omega, const int num, const int num_of_tasks, Dtype* loss) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      int i = index / num;
      int j = index % num;
      int p = task_index[i];
      int q = task_index[j];
      
      loss[index] = kernel[index] * Omega[p * num_of_tasks + q];
  }
}

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    //flatten data
    int count = 0;
    for(int i = 0;i < bottom.size();++i){
        const Dtype* bottom_data = bottom[i]->gpu_data();
        caffe_gpu_memcpy(sizeof(Dtype) * bottom[i]->count(), bottom_data, data_.mutable_gpu_data() + count);
        count += bottom[i]->count();
    }

    //calculate square distance
    int nthreads = total_W_num_ * total_W_num_ * dimension_;
    CalculateKernelGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, data_.gpu_data(), data_.mutable_gpu_diff(),
      total_W_num_, dimension_);
    
    //use redundent memory of data as (1, 1, ..., 1) multiplier
    Dtype* vector_sum_multiplier = data_.mutable_gpu_data() + total_W_num_ * dimension_;
    CHECK_GE((total_W_num_ - 1) * total_W_num_ * dimension_, dimension_) << "not enough data space";
    caffe_gpu_set(dimension_, Dtype(1.0), vector_sum_multiplier);
    
    caffe_gpu_gemv(CblasNoTrans, total_W_num_ * total_W_num_, dimension_, Dtype(1.0), 
            data_.gpu_diff(), vector_sum_multiplier, Dtype(1.0), kernel_.mutable_gpu_data());

    //calculate sigma with square distance
    caffe_gpu_asum(total_W_num_ * total_W_num_, kernel_.gpu_data(), &sigma_);
    Dtype kernel_coefficient = -0.5 * total_W_num_ * (total_W_num_ - 1)  / sigma_;

    caffe_gpu_scal(total_W_num_ * total_W_num_, kernel_coefficient, kernel_.mutable_gpu_data());
    caffe_gpu_exp(total_W_num_ * total_W_num_, kernel_.gpu_data(), kernel_.mutable_gpu_data());
    
    Dtype* task_index = temp_.mutable_cpu_data();
    int index = 0;
    for(int i = 0, j = 0;i < total_W_num_;++i, ++j){
        if(j >= D_.cpu_data()[index]){
            j = 0;
            index++;
        }
        task_index[i] = index;
    }
    
    nthreads = total_W_num_ * total_W_num_;
    CalculateLossGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, kernel_.gpu_data(), temp_.gpu_data(),
      Omega_.gpu_data(), total_W_num_, num_of_tasks_, kernel_.mutable_gpu_diff());
    
    Dtype loss = 0;
    caffe_gpu_set(total_W_num_ * total_W_num_, Dtype(1.0), vector_sum_multiplier);
    CHECK_GE((total_W_num_ - 1) * total_W_num_ * dimension_, total_W_num_ * total_W_num_) << "not enough data space";
    caffe_gpu_dot(total_W_num_ * total_W_num_, vector_sum_multiplier, kernel_.gpu_diff(), &loss);
    
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void CalculateDiffGPU(const int nthreads,
          const Dtype* data, const Dtype* Omega, const Dtype* task_index, 
          const Dtype* kernel, const Dtype coefficient, const int num, 
          const int dim, const int num_of_tasks, Dtype* diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
      int i = index / dim / num;
      int j = index / dim % num;
      int k = index % dim;
      int p = task_index[i];
      int q = task_index[j];
      
      Dtype omega = Omega[p * num_of_tasks + q];
      
      diff[(i * num + j) * dim + k] = 2 * omega * kernel[i * num + j] * coefficient * (data[(i * num + j) * dim + k] - data[(j * num + i) * dim + k]);
  }
}

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype kernel_coefficient = -0.5 * total_W_num_ * (total_W_num_ - 1) / sigma_;

    Dtype* task_index = temp_.mutable_cpu_data();
    
    //calculate diff
    int nthreads = total_W_num_ * total_W_num_ * dimension_;
    CalculateDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, data_.gpu_data(), Omega_.gpu_data(),
      temp_.gpu_data(), kernel_.gpu_data(), kernel_coefficient, total_W_num_, dimension_, 
      num_of_tasks_, data_.mutable_gpu_diff());
    //add diff to bottom diff
    //use redundent memory of data as (1, 1, ..., 1) multiplier
    Dtype* vector_sum_multiplier = data_.mutable_gpu_data() + total_W_num_ * dimension_;
    caffe_gpu_set(total_W_num_, Dtype(1.0), vector_sum_multiplier);
    int offset = 0;
    for(int i = 0;i < num_of_tasks_;++i){
        for(int j = 0;j < D_.cpu_data()[i];++j){
            caffe_gpu_gemv(CblasTrans, total_W_num_, dimension_, Dtype(1.0), 
                data_.gpu_diff() + offset, vector_sum_multiplier, Dtype(1.0), bottom[i]->mutable_gpu_diff() + j * dimension_);
            offset += total_W_num_ * dimension_;
        }
        //scale by loss_weight
        caffe_gpu_scal(D_.cpu_data()[i] * dimension_, top[0]->cpu_diff()[0], bottom[i]->mutable_gpu_diff());
    }
    
    //update Omega
    caffe_gpu_set(num_of_tasks_ * num_of_tasks_, Dtype(0), Omega_.mutable_gpu_diff());
    Dtype* A = Omega_.mutable_cpu_diff();
    const Dtype* kernel = kernel_.cpu_data();
    
    for(int i = 0;i < (total_W_num_ * total_W_num_);++i){
        int p = task_index[i / total_W_num_];
        int q = task_index[i % total_W_num_];
        A[p * num_of_tasks_ + q] += kernel[i];
    }
    
    caffe_cpu_matrix_sqrt(num_of_tasks_, A);
    //calculate trace
    Dtype trace = 0;
    for(int i = 0;i < num_of_tasks_;++i){
        trace += A[i * (num_of_tasks_ + 1)];
    }
    //divide by trace
    caffe_scal(num_of_tasks_ * num_of_tasks_, 1 / trace, A);
    //inverse
    caffe_cpu_inverse(num_of_tasks_, A);
    //copy to Omega
    caffe_gpu_memcpy(sizeof(Dtype) * num_of_tasks_ * num_of_tasks_, A, Omega_.mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiTaskWeightLossLayer);

}  // namespace caffe
