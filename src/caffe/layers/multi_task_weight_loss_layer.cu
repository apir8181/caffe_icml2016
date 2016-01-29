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
__global__ void calculate_pairwise_sqr_distance(
	  const Dtype* data, Dtype *out, 
		int num_classes, int feature_dim) {
	
	CUDA_KERNEL_LOOP(ij, num_classes * num_classes) {
		int i = ij / num_classes, j = ij % num_classes;
		Dtype val = 0;
		for (int d = 0; d < feature_dim; ++ d) {
			Dtype diff = data[i * feature_dim + d] - data[j * feature_dim + d];
			val += diff * diff;
		}
		out[ij] = val;
	}

}

template <typename Dtype>
__global__ void calculate_pairwise_kernel(
	  const Dtype* pairwise_sqr_distance, Dtype *out,
		Dtype sigma, int count) {
	
	CUDA_KERNEL_LOOP(i, count) {
		Dtype factor = -1.0 / (2 * sigma);
		out[i] = exp(factor * pairwise_sqr_distance[i]);
	}

}


template <typename Dtype>
__global__ void calculate_loss(
	  const Dtype* pairwise_kernel, const Dtype* Omega,
		const int* task_start_index, const int* task_end_index,	Dtype* out, 
		int num_tasks, int num_classes) {
	
	CUDA_KERNEL_LOOP(ij, num_tasks * num_tasks) {
		int i = ij / num_tasks, j = ij % num_tasks;
		Dtype val = 0;
		for (int c1 = task_start_index[i]; c1 < task_end_index[i]; ++ c1)
			for (int c2 = task_start_index[j]; c2 < task_end_index[j]; ++ c2) {
				int idx = c1 * num_classes + c2;
				val += Omega[ij] * pairwise_kernel[idx];
			}
		out[ij] = val;
	}

}


template <typename Dtype>
__global__ void calculate_A(
	  const Dtype* pairwise_kernel, const int* task_start_index, 
		const int* task_end_index, Dtype* out, 
		int num_tasks, int num_classes) {
	
	CUDA_KERNEL_LOOP(ij, num_tasks * num_tasks) {
		int i = ij / num_tasks, j = ij % num_tasks;
		Dtype val = 0;
		for (int c1 = task_start_index[i]; c1 < task_end_index[i]; ++ c1)
			for (int c2 = task_start_index[j]; c2 < task_end_index[j]; ++ c2) {
				int idx = c1 * num_classes + c2;
				val += pairwise_kernel[idx];
			}
		out[ij] = val;
	}

}
	

template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int* task_start_index = task_start_index_.gpu_data();
	const int* task_end_index = task_end_index_.gpu_data();
	const Dtype* Omega = this->blobs_[0]->gpu_data();
	Dtype* data = data_.mutable_gpu_data();
	Dtype* pairwise_sqr_distance = pairwise_sqr_distance_.mutable_gpu_data();
	Dtype* pairwise_kernel = pairwise_kernel_.mutable_gpu_data();
	Dtype* loss = loss_.mutable_gpu_data();

	//flatten data
	int count = 0;
	for(int i = 0; i < bottom.size(); ++ i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		caffe_gpu_memcpy(sizeof(Dtype) * bottom[i]->count(), bottom_data, 
										 data + count);
		count += bottom[i]->count();
	}

	// calculate distance
	calculate_pairwise_sqr_distance<Dtype><<<CAFFE_GET_BLOCKS(num_classes_ * num_classes_),
		CAFFE_CUDA_NUM_THREADS>>>(data, pairwise_sqr_distance, num_classes_, feature_dim_);

	if(debug_detail_){
		LOG(INFO) << "-------------------------------------distance matrix";
		print_gpu_matrix(pairwise_sqr_distance, num_classes_, num_classes_, num_classes_, num_classes_);
	}

	// calculate sigma
	caffe_gpu_asum<Dtype>(pairwise_sqr_distance_.count(), pairwise_sqr_distance, &sigma_);
	sigma_ /= num_classes_ * num_classes_;

	if (debug_info_) {
		LOG(INFO) << "sigma:" << sigma_;
	}

	// calculate kernel
	calculate_pairwise_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_classes_ * num_classes_),
		CAFFE_CUDA_NUM_THREADS>>>(pairwise_sqr_distance, pairwise_kernel, sigma_, num_classes_ * num_classes_);

	if (debug_detail_) {
		LOG(INFO) << "--------------------------------------kernel matrix";
		print_gpu_matrix(pairwise_kernel, num_classes_, num_classes_, num_classes_, num_classes_);
	}

	// calculate A
	calculate_A<Dtype><<<CAFFE_GET_BLOCKS(num_tasks_ * num_tasks_),
		CAFFE_CUDA_NUM_THREADS>>>(pairwise_kernel, task_start_index, task_end_index,
															A_.mutable_gpu_data(), num_tasks_, num_classes_);

	if(debug_info_){
		LOG(INFO) << "-------------------------------------A";
		print_gpu_matrix(A_.gpu_data(), num_tasks_, num_tasks_, num_tasks_, num_tasks_);
	}


	// calculate loss
	calculate_loss<Dtype><<<CAFFE_GET_BLOCKS(num_tasks_ * num_tasks_),
		CAFFE_CUDA_NUM_THREADS>>>(pairwise_kernel, Omega, 
															task_start_index, task_end_index, loss,
															num_tasks_, num_classes_);

	if(debug_info_){
		LOG(INFO) << "-------------------------------------Omega";
		print_gpu_matrix(Omega, num_tasks_, num_tasks_, num_tasks_, num_tasks_);

		LOG(INFO) << "-------------------------------------loss matrix";
		print_gpu_matrix(loss, num_tasks_, num_tasks_, num_tasks_, num_tasks_);
	}

	
	Dtype sum_loss = 0;
	for (int i = 0; i < loss_.count(); ++ i)
		sum_loss += loss_.cpu_data()[i];
	top[0]->mutable_cpu_data()[0] = sum_loss;
}


template <typename Dtype>
__global__ void backward_weight(
	  const Dtype* pairwise_kernel, const Dtype* data, const int* data2task,
	  const Dtype* Omega, const Dtype sigma, Dtype* out,
		int num_tasks, int num_classes, int feature_dim) {
	
	CUDA_KERNEL_LOOP(id, num_classes * feature_dim) {
		int i = id / feature_dim, d = id % feature_dim;
		Dtype val = 0;
		for (int j = 0; j < num_classes; ++ j) {
			int ij = i * num_classes + j;
			int jd = j * feature_dim + d;
			int task_i = data2task[i], task_j = data2task[j];
			int task_ij = task_i * num_tasks + task_j;
			Dtype weight = - (data[id] - data[jd]) / sigma;
			if (task_i != task_j) {
				val += Omega[task_ij] * pairwise_kernel[ij] * weight;
			}
		}
		out[id] = val;
	}

}


template <typename Dtype>
void MultiTaskWeightLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* data = data_.gpu_data();
	const Dtype* pairwise_kernel = pairwise_kernel_.gpu_data();
	const int* task_start_index = task_start_index_.gpu_data();
	const int* task_end_index = task_end_index_.gpu_data();
	const int* data2task = data2task_.gpu_data();
	Dtype* Omega = this->blobs_[0]->mutable_gpu_data();
	Dtype* W_diff = data_.mutable_gpu_diff();

	caffe_cpu_matrix_sqrt(num_tasks_, A_.mutable_cpu_data());

  if(debug_info_){
		LOG(INFO) << "-------------------------------------A^0.5";
		print_gpu_matrix(A_.gpu_data(), num_tasks_, num_tasks_, num_tasks_, num_tasks_);    
	}

  // calculate trace(A)
  Dtype trace = 0;
  for (int i = 0; i < num_tasks_; ++ i) {
		trace += A_.mutable_cpu_data()[i * (num_tasks_ + 1)];
	}

  // divide A by trace(A)
  caffe_gpu_scal<Dtype>(num_tasks_ * num_tasks_, 1 / trace, A_.mutable_gpu_data());
    
  if(debug_info_){
		LOG(INFO) << "-------------------------------------A^0.5/trace(A^0.5)";
		print_gpu_matrix(A_.gpu_data(), num_tasks_, num_tasks_, num_tasks_, num_tasks_);
  }

  for (int i = 0; i < num_tasks_; ++ i) {
		A_.mutable_cpu_data()[i * (num_tasks_ + 1)] += 1e-5;
	}

  // inverse A
  caffe_cpu_inverse<Dtype>(num_tasks_, A_.mutable_cpu_data());
    
  if(debug_info_){
		LOG(INFO) << "-------------------------------------Omega_new";
		print_gpu_matrix(A_.gpu_data(), num_tasks_, num_tasks_, num_tasks_, num_tasks_);    
  }
  
  // copy to Omega
  caffe_gpu_memcpy(sizeof(Dtype) * num_tasks_ * num_tasks_, 
									 A_.gpu_data(), Omega);

	
	// set weight gradient
	backward_weight<Dtype><<<CAFFE_GET_BLOCKS(num_classes_ * feature_dim_),
		CAFFE_CUDA_NUM_THREADS>>>(pairwise_kernel, data, data2task,
															Omega, sigma_, W_diff,
															num_tasks_, num_classes_, feature_dim_);

	// backward gradient to bottom
	int count = 0;
	for (int i = 0; i < num_tasks_; ++ i) {
		Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
		caffe_gpu_memcpy(bottom[i]->count() * sizeof(Dtype), 
										 W_diff + count, bottom_diff);
		count += bottom[i]->count();
		caffe_gpu_scal<Dtype>(bottom[i]->count(), top[0]->cpu_diff()[0], bottom_diff);
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(MultiTaskWeightLossLayer);

}  // namespace caffe
