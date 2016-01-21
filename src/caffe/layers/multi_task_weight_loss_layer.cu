#include <algorithm>
#include <cfloat>
#include <vector>
#include <climits>

#include "caffe/layers/multi_task_weight_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype RBF_kernel(const Dtype* X, const Dtype* Y, Dtype* temp, Dtype coefficient, int dim){
    Dtype square_distance;
    caffe_gpu_sub(dim, X, Y, temp);
    caffe_gpu_dot(dim, temp, temp, &square_distance);
    return exp(coefficient * square_distance);
}

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Dtype* tempW1 = temp_.mutable_gpu_data();
    Dtype* tempW2 = temp_.mutable_gpu_diff();
    
    //calculate sigma in RBF kernel
    srand((unsigned int) time(0));
    Dtype square_distance;
    sigma_ = 0;
    for(int i = 0;i < total_W_num_;++i){
        //random sample two W
        int w1 = rand() % total_W_num_;
        int w2 = rand() % total_W_num_;
        w2 = (w1 == w2) ? (w1 + 1) % total_W_num_ : w2;
        for(int j = 0;j < N_.size();++j){
            if(w1 < N_[i]){
                caffe_gpu_memcpy(K_[0] * sizeof(Dtype), bottom[i]->gpu_data() + w1 * K_[0], tempW1);
                w1 = INT_MAX;
            }
            else{
                w1 -= N_[i];
            }
            if(w2 < N_[i]){
                caffe_gpu_memcpy(K_[0] * sizeof(Dtype), bottom[i]->gpu_data() + w2 * K_[0], tempW2);
                w1 = INT_MAX;
            }
            else{
                w2 -= N_[i];
            }
        }
        
        caffe_gpu_sub<Dtype>(K_[0], tempW1, tempW2, tempW2);
        caffe_gpu_dot<Dtype>(K_[0], tempW2, tempW2, &square_distance);
        sigma_ += square_distance;
    }

    //coefficient in RBF kernel
    Dtype kernel_coefficient = -0.5 * total_W_num_ / sigma_;
    
    Dtype loss = 0;
    for(int t1 = 0;t1 < bottom.size();++t1){
        for(int t2 = 0;t2 < bottom.size();++t2){
            Dtype omega = Omega_[t1 * bottom.size() + t2];
            for(int c1 = 0;c1 < N_[t1];++c1){
                for(int c2 = 0;c2 < N_[t2];++c2){
                    loss += omega * RBF_kernel(bottom[t1]->gpu_data() + c1 * K_[0],
                                               bottom[t2]->gpu_data() + c2 * K_[0],
                                               tempW1, kernel_coefficient, K_[0]);
                }
            }
        }
    }
    
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype kernel_coefficient = -0.5 * total_W_num_ / sigma_;
    Dtype* tempW1 = temp_.mutable_gpu_data();
    Dtype* tempW2 = temp_.mutable_gpu_diff();

    caffe_set(num_of_tasks_ * num_of_tasks_, Dtype(0), Omega_cache_);
    //update W
    for(int t1 = 0;t1 < bottom.size();++t1){
        for(int t2 = 0;t2 < bottom.size();++t2){
            Dtype omega = Omega_[t1 * bottom.size() + t2];
            for(int c1 = 0;c1 < N_[t1];++c1){
                for(int c2 = 0;c2 < N_[t2];++c2){
                    Dtype kernel = RBF_kernel(bottom[t1]->gpu_data() + c1 * K_[0],
                                              bottom[t2]->gpu_data() + c2 * K_[0],
                                              tempW1, kernel_coefficient, K_[0]);
                    Omega_cache_[t1 * num_of_tasks_ + t2] += kernel;
                    Dtype diff_coefficient = omega * kernel * 2 * kernel_coefficient * top[0]->cpu_diff()[0];
                    caffe_gpu_sub(K_[0], bottom[t1]->gpu_data() + c1 * K_[0], bottom[t2]->gpu_data() + c2 * K_[0], tempW1);
                    caffe_gpu_axpby(K_[0], diff_coefficient, tempW1, Dtype(0), bottom[t1]->mutable_gpu_diff() + c1 * K_[0]);
                }
            }
        }
    }
    
    //update Omega
    caffe_cpu_matrix_sqrt(num_of_tasks_, Omega_cache_);
    //calculate trace
    Dtype trace = 0;
    for(int i = 0;i < num_of_tasks_;++i){
        trace += Omega_cache_[i * (num_of_tasks_ + 1)];
    }
    //divide by trace
    caffe_scal(num_of_tasks_ * num_of_tasks_, 1 / trace, Omega_cache_);
    //inverse
    caffe_cpu_inverse(num_of_tasks_, Omega_cache_);
    //copy to Omega
    caffe_copy(num_of_tasks_ * num_of_tasks_, Omega_cache_, Omega_);
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiTaskWeightLossLayer);

}  // namespace caffe
