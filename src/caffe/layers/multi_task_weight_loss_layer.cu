#include <algorithm>
#include <cfloat>
#include <vector>
#include <climits>
#include <math.h>

#include "caffe/layers/multi_task_weight_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output_matrix.hpp"

namespace caffe {

  
template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Forward_gpu_FillWByTask(
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < num_tasks_; ++ i) {
    const Dtype* src = bottom[i]->gpu_data();
    Dtype* dst = W_by_task_[i]->mutable_gpu_data();
    caffe_gpu_memcpy<Dtype>(bottom[i]->count() * sizeof(Dtype), src, dst);
  }
}

  
template <typename Dtype>
void MultiTaskWeightLossLayer::Forward_gpu_FillAByTask() {
  for (int i = 0; i < num_task_; ++ i) {
    for (int j = 0; j < num_task_; ++ j) {
      const Dtype* W1 = W_by_task[i]->gpu_data();
      const Dtype* W2 = W_by_task[j]->gpu_data();
      const Dtype* Theta = Theta_.gpu_data();
      Dtype* temp_D_C = temp_D_C_.mutable_gpu_data();
      Dtype* temp_D_D = temp_D_D_.muteable_gpu_data();
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                            num_feature_, num_class_, num_class_,
                            1, W1, Theta, 0, temp_D_C);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            num_feature_, num_feature_, num_class_,
                            1, temp_D_C, W2, 0, temp_D_D);
      Dtype trace;
      caffe_gpu_trace<Dtype>(num_feature_, temp_D_D, &trace);
      int ij = i * num_task_ + j;
      A_by_task_.mutable_cpu_data()[ij] = trace;
    }
  }
}


template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Forward_gpu_FillWByTask(bottom);
  Forward_gpu_FillAByTask();

  const Dtype* Omega = Omega_.cpu_data();
  const Dtype* A_by_task = A_by_task_.cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < num_task_ * num_task_; ++ i) {
    loss += Omega[i] * A_by_task[i];
  }
  top[0]->mutable_cpu_data[0] = loss;
}


template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Backward_gpu_UpdateOmega() {
  if(debug_info_){
		LOG(INFO) << "--------------------A_tt";
		print_gpu_matrix(A_by_task_.gpu_data(), num_task_, num_task_,
                     num_task_, num_task_);
	}

	caffe_cpu_matrix_sqrt(num_task_, A_by_task_.mutable_cpu_data());

  if(debug_info_){
		LOG(INFO) << "--------------------A_tt^0.5";
		print_gpu_matrix(A_by_task_.gpu_data(), num_task_, num_task_,
                     num_task_, num_task_);    
	}

  Dtype trace;
  caffe_gpu_trace<Dtype>(num_task_, A_by_task_.gpu_data(), &trace);
  caffe_gpu_scal<Dtype>(A_by_task_.count(), 1 / trace,
                        A_by_task_.mutable_gpu_data());

  if(debug_info_){
		LOG(INFO) << "--------------------A_tt^0.5/trace(A_tt^0.5)";
		print_gpu_matrix(A_by_task_.gpu_data(), num_task_, num_task_,
                     num_task_, num_task_);    
	}
  
  caffe_cpu_inverse<Dtype>(num_task_, A_by_task_.mutable_cpu_data());
  
  if(debug_info_){
		LOG(INFO) << "--------------------Omega_new";
		print_gpu_matrix(A_by_task_.gpu_data(), num_task_, num_task_,
                     num_task_, num_task_);    
	}

  caffe_gpu_memcpy(A_by_task_.count() * sizeof(Dtype),
                   A_by_task_.gpu_data(), Omega_.mutable_gpu_data());
}
  
template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Backward_gpu_FillWByClass(
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < num_class_; ++ i) {
    for (int j = 0; j < num_tasks_; ++ j) {
      const Dtype* src = bottom[j]->gpu_data() + i * num_feature_;
      Dtype* dst = W_by_class_[i]->mutable_gpu_data() + j * num_feature_;
      caffe_gpu_memcpy<Dtype>(num_feature_ * sizeof(Dtype), src, dst);
    }
  }
}


template <typename Dtype>
void MultiTaskWeightLossLayer::Backward_gpu_FillAByClass() {
  for (int i = 0; i < num_class_; ++ i) {
    for (int j = 0; j < num_class_; ++ j) {
      const Dtype* W1 = W_by_class_[i]->gpu_data();
      const Dtype* W2 = W_by_class_[j]->gpu_data();
      const Dtype* Omega = Omega_.mutable_gpu_data();
      Dtype* temp_D_T = temp_D_T_.mutable_gpu_data();
      Dtype* temp_D_D = temp_D_D_.muteable_gpu_data();
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                            num_feature_, num_task_, num_task_,
                            1, W1, Omega, 0, temp_D_T);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            num_feature_, num_feature_, num_class_,
                            1, temp_D_T, W2, 0, temp_D_D);
      Dtype trace;
      caffe_gpu_trace<Dtype>(num_feature_, temp_D_D, &trace);
      int ij = i * num_class_ + j;
      A_by_class_.mutable_cpu_data()[ij] = trace;
    }
  }
}


template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Backward_gpu_UpdateTheta() {
  if(debug_info_){
		LOG(INFO) << "--------------------A_cc";
		print_gpu_matrix(A_by_class_.gpu_data(), num_class_, num_class_,
                     num_class_, num_class_);    
	}

	caffe_cpu_matrix_sqrt(num_class_, A_by_class_.mutable_cpu_data());

  if(debug_info_){
		LOG(INFO) << "--------------------A_cc^0.5";
		print_gpu_matrix(A_by_class_.gpu_data(), num_class_, num_class_,
                     num_class_, num_class_);    
	}

  Dtype trace;
  caffe_gpu_trace<Dtype>(num_class_, A_by_class_.gpu_data(), &trace);
  caffe_gpu_scal<Dtype>(A_by_class_.count(), 1 / trace,
                        A_by_class_.mutable_gpu_data());

  if(debug_info_){
		LOG(INFO) << "--------------------A_cc^0.5/trace(A_cc^0.5)";
		print_gpu_matrix(A_by_class_.gpu_data(), num_class_, num_class_,
                     num_class_, num_class_);
	}
  
  caffe_cpu_inverse<Dtype>(num_class_, A_by_class_.mutable_cpu_data());
  
  if(debug_info_){
		LOG(INFO) << "--------------------Theta_new";
		print_gpu_matrix(A_by_class_.gpu_data(), num_class_, num_class_,
                     num_class_, num_class_);
	}

  caffe_gpu_memcpy(A_by_class_.count() * sizeof(Dtype),
                   A_by_class_.gpu_data(), Theta_.mutable_gpu_data());
}


template <typename Dtype>
__global__ void MultiTaskWeightLossLayer::Backward_gpu_Backprop(
  const vector<Blob<Dtype>*>& bottom, const vecotr<Blob<Dtype>*>& top) {
  const Dtype* Omega = Omega_.gpu_data();
  const Dtype* Theta = Theta_.gpu_data();
  
  for (int i = 0; i < num_task_; ++ i) {
    Dtype* diff = bottom[i]->mutable_gpu_diff();
    for (int j = 0; j < num_task_; ++ j) {
      const int ij = i * num_task_ + j;
      const Dtype* W = W_by_task_[j]->gpu_data();
      const Dtype weight = 2 * top[0]->gpu_diff() * Omega[ij];
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                            num_class_, num_feature_, num_class_,
                            weight, Theta, W, 1, diff);
    }
  }
}


template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!fronzen_Omega_) {
    Backward_gpu_UpdateOmega();
  }
  Backward_gpu_FillWByClass(bottom);
  Backward_gpu_FillAByClass();
  if (!fronzen_Theta_) {
    Backward_gpu_UpdateTheta();
  }
  Backward_gpu_Backprop(bottom, top);
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiTaskWeightLossLayer);

}  // namespace caffe
