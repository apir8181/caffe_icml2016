#include <algorithm>
#include <cfloat>
#include <vector>
#include <climits>
#include <math.h>

#include "caffe/layers/multi_task_weight_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/output_matrix.hpp"

namespace caffe {

// template <typename Dtype>
// __global__ void CalculateKernelGPU(const int nthreads,
//           const Dtype* data, Dtype* square_diff, int num, int dim) {
//   CUDA_KERNEL_LOOP(index, nthreads) {
//       int i = index / dim / num;
//       int j = index / dim % num;
//       int k = index % dim;
      
//       Dtype diff = data[i * dim + k] - data[j * dim + k];
//       square_diff[(i * num + j) * dim + k] = diff * diff;
//   }
// }

// template <typename Dtype>
// __global__ void CalculateLossGPU(const int nthreads,
//           const Dtype* kernel, const Dtype* task_index, 
//           const Dtype* Omega, const int num, const int num_of_tasks, Dtype* loss) {
//   CUDA_KERNEL_LOOP(index, nthreads) {
//       int i = index / num;
//       int j = index % num;
//       int p = task_index[i];
//       int q = task_index[j];
      
//       loss[index] = kernel[index] * Omega[p * num_of_tasks + q];
//   }
// }

template <typename Dtype>
__global__ void calculate_pairwise_distance(
	  const Dtype* data, Dtype *out, 
		int num_classes, int feature_dim) {
	
	CUDA_KERNEL_LOOP(ij, num_classes * num_classes) {
		int i = ij / num_classes, j = ij % num_classes;
		Dtype val = 0;
		for (int d = 0; d < feature_dim; ++ d) {
			Dtype diff = data[i * feature_dim + d] - data[j * feature_dim + d];
			val += diff * diff;
		}
		out[ij] = sqrt(val);
	}

}

template <typename Dtype>
__global__ void calculate_pairwise_kernel(
	  const Dtype* pairwise_distance, Dtype *out,
		Dtype sigma, int count) {
	
	CUDA_KERNEL_LOOP(i, count) {
		Dtype val = pairwise_distance[i] * pairwise_distance[i];
		Dtype factor = -1.0 / (2 * sigma * sigma);
		out[i] = exp(factor * val);
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
void MultiTaskWeightLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int* task_start_index = task_start_index_.gpu_data();
	const int* task_end_index = task_end_index_.gpu_data();
	const Dtype* Omega = Omega_.gpu_data();
	Dtype* data = data_.mutable_gpu_data();
	Dtype* pairwise_distance = pairwise_distance_.mutable_gpu_data();
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
	int num_threads = num_classes_ * num_classes_;
	calculate_pairwise_distance<Dtype><<<CAFFE_GET_BLOCKS(num_threads),
		CAFFE_CUDA_NUM_THREADS>>>(data, pairwise_distance, num_classes_, feature_dim_);

	// calculate sigma
	caffe_gpu_asum<Dtype>(pairwise_distance_.count(), pairwise_distance, &sigma_);
	sigma_ /= num_classes_;

	if (debug_info_) {
		LOG(INFO) << "sigma:" << sigma_;
	}

	// calculate kernel
	calculate_pairwise_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_threads),
		CAFFE_CUDA_NUM_THREADS>>>(pairwise_distance, pairwise_kernel, sigma_, num_threads);

	// calculate loss
	calculate_loss<Dtype><<<CAFFE_GET_BLOCKS(num_tasks_ * num_tasks_),
		CAFFE_CUDA_NUM_THREADS>>>(pairwise_kernel, Omega, 
															task_start_index, task_end_index, loss,
															num_tasks_, num_classes_);

	if(debug_info_){
		LOG(INFO) << "-------------------------------------loss matrix";
		print_gpu_matrix(loss, num_tasks_, num_tasks_, num_tasks_, num_tasks_);
	}

	
	Dtype sum_loss;
	caffe_gpu_asum<Dtype>(loss_.count(), loss, &sum_loss);
	top[0]->mutable_cpu_data()[0] = sum_loss;

    //calculate square distance
    // int nthreads = total_W_num_ * total_W_num_ * dimension_;
    // CalculateKernelGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    //   CAFFE_CUDA_NUM_THREADS>>>(nthreads, data_.gpu_data(), data_.mutable_gpu_diff(),
    //   total_W_num_, dimension_);
    
    // //use redundent memory of data as (1, 1, ..., 1) multiplier
    // Dtype* vector_sum_multiplier = data_.mutable_gpu_data() + total_W_num_ * dimension_;
    // CHECK_GE((total_W_num_ - 1) * total_W_num_ * dimension_, dimension_) << "not enough data space";
    // caffe_gpu_set(dimension_, Dtype(1.0), vector_sum_multiplier);
    
    // caffe_gpu_gemv(CblasNoTrans, total_W_num_ * total_W_num_, dimension_, Dtype(1.0), 
    //         data_.gpu_diff(), vector_sum_multiplier, Dtype(1.0), kernel_.mutable_gpu_data());

    // //calculate sigma with square distance
    // caffe_gpu_asum(total_W_num_ * total_W_num_, kernel_.gpu_data(), &sigma_);
    // Dtype kernel_coefficient = -0.5 * total_W_num_ * (total_W_num_ - 1)  / sigma_;

    // caffe_gpu_scal(total_W_num_ * total_W_num_, kernel_coefficient, kernel_.mutable_gpu_data());
    // caffe_gpu_exp(total_W_num_ * total_W_num_, kernel_.gpu_data(), kernel_.mutable_gpu_data());
    
    // Dtype* task_index = temp_.mutable_cpu_data();
    // int index = 0;
    // for(int i = 0, j = 0;i < total_W_num_;++i, ++j){
    //     if(j >= D_.cpu_data()[index]){
    //         j = 0;
    //         index++;
    //     }
    //     task_index[i] = index;
    // }
    
    // nthreads = total_W_num_ * total_W_num_;
    // CalculateLossGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    //   CAFFE_CUDA_NUM_THREADS>>>(nthreads, kernel_.gpu_data(), temp_.gpu_data(),
    //   Omega_.gpu_data(), total_W_num_, num_of_tasks_, kernel_.mutable_gpu_diff());
    
    // Dtype loss = 0;
    // caffe_gpu_set(total_W_num_ * total_W_num_, Dtype(1.0), vector_sum_multiplier);
    // CHECK_GE((total_W_num_ - 1) * total_W_num_ * dimension_, total_W_num_ * total_W_num_) << "not enough data space";
    // caffe_gpu_dot(total_W_num_ * total_W_num_, vector_sum_multiplier, kernel_.gpu_diff(), &loss);
    
    // top[0]->mutable_cpu_data()[0] = loss;
}

// template <typename Dtype>
// __global__ void CalculateDiffGPU(const int nthreads,
//           const Dtype* data, const Dtype* Omega, const Dtype* task_index, 
//           const Dtype* kernel, const Dtype coefficient, const int num, 
//           const int dim, const int num_of_tasks, Dtype* diff) {
//   CUDA_KERNEL_LOOP(index, nthreads) {
//       int i = index / dim / num;
//       int j = index / dim % num;
//       int k = index % dim;
//       int p = task_index[i];
//       int q = task_index[j];
      
//       Dtype omega = Omega[p * num_of_tasks + q];
      
//       diff[(i * num + j) * dim + k] = 2 * omega * kernel[i * num + j] * coefficient * (data[i * dim + k] - data[j * dim + k]);
//   }
// }


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
			Dtype weight = - (data[id] - data[jd]) / (sigma * sigma);
			val += Omega[task_ij] * pairwise_kernel[ij] * weight;
		}
		out[id] = val;
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
void MultiTaskWeightLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* data = data_.gpu_data();
	const Dtype* pairwise_kernel = pairwise_kernel_.gpu_data();
	const int* task_start_index = task_start_index_.gpu_data();
	const int* task_end_index = task_end_index_.gpu_data();
	const int* data2task = data2task_.gpu_data();
	Dtype* Omega = Omega_.mutable_gpu_data();
	Dtype* W_diff = data_.mutable_gpu_diff();
	
	// set weight gradient
	backward_weight<Dtype><<<CAFFE_GET_BLOCKS(num_classes_ * feature_dim_),
		CAFFE_CUDA_NUM_THREADS>>>(pairwise_kernel, data, data2task,
															Omega, sigma_, W_diff,
															num_tasks_, num_classes_, feature_dim_);

	// backward gradient to bottom
	for (int i = 0; i < num_tasks_; ++ i) {
		Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
		int start_index = task_start_index_.cpu_data()[i];
		int end_index = task_end_index_.cpu_data()[i];
		int count = (end_index - start_index) * feature_dim_;
		caffe_gpu_memcpy(count * sizeof(Dtype), 
										 W_diff + start_index * feature_dim_, bottom_diff);
		caffe_gpu_scal(count, top[0]->cpu_diff()[0], bottom_diff);
	}

	// calculate A
	calculate_A<Dtype><<<CAFFE_GET_BLOCKS(num_tasks_ * num_tasks_),
		CAFFE_CUDA_NUM_THREADS>>>(pairwise_kernel, task_start_index, task_end_index,
															A_.mutable_gpu_data(), num_tasks_, num_classes_);

	if(debug_info_){
		LOG(INFO) << "-------------------------------------begin";
		print_gpu_matrix(A_.gpu_data(), num_tasks_, num_tasks_, num_tasks_, num_tasks_);
	}

	caffe_cpu_matrix_sqrt(num_tasks_, A_.mutable_cpu_data());

  if(debug_info_){
		LOG(INFO) << "-------------------------------------aftar sqrt";
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
		LOG(INFO) << "-------------------------------------aftar divide trace";
		print_gpu_matrix(A_.gpu_data(), num_tasks_, num_tasks_, num_tasks_, num_tasks_);
  }

  // inverse A
  caffe_cpu_inverse<Dtype>(num_tasks_, A_.mutable_cpu_data());
    
  if(debug_info_){
		LOG(INFO) << "-------------------------------------aftar inverse";
		print_gpu_matrix(A_.gpu_data(), num_tasks_, num_tasks_, num_tasks_, num_tasks_);    
  }
  
  // copy to Omega
  caffe_gpu_memcpy(sizeof(Dtype) * num_tasks_ * num_tasks_, 
									 A_.gpu_data(), Omega);

    // Dtype kernel_coefficient = -0.5 * total_W_num_ * (total_W_num_ - 1) / sigma_;

    // Dtype* task_index = temp_.mutable_cpu_data();
    
    // //calculate diff
    // int nthreads = total_W_num_ * total_W_num_ * dimension_;
    // CalculateDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
    //   CAFFE_CUDA_NUM_THREADS>>>(nthreads, data_.gpu_data(), Omega_.gpu_data(),
    //   temp_.gpu_data(), kernel_.gpu_data(), kernel_coefficient, total_W_num_, dimension_, 
    //   num_of_tasks_, data_.mutable_gpu_diff());
    // //add diff to bottom diff
    // //use redundent memory of data as (1, 1, ..., 1) multiplier
    // Dtype* vector_sum_multiplier = data_.mutable_gpu_data() + total_W_num_ * dimension_;
    // caffe_gpu_set(total_W_num_, Dtype(1.0), vector_sum_multiplier);
    
    // for(int i = 0;i < num_of_tasks_;++i){
    //     for(int j = 0;j < D_.cpu_data()[i];++j){
    //         caffe_gpu_set(dimension_, Dtype(0), bottom[i]->mutable_gpu_diff() + j * dimension_);
    //         for(int k = 0;k < num_of_tasks_;++k){
    //             if(i == k) continue;
    //             int offset = ((i * D_.cpu_data()[0] + j) * total_W_num_ + k * D_.cpu_data()[0] + j) * dimension_;
    //             caffe_gpu_add(dimension_, data_.gpu_diff() + offset, bottom[i]->gpu_diff() + j * dimension_, bottom[i]->mutable_gpu_diff() + j * dimension_);
    //         }
    //     }
    //     caffe_gpu_scal(D_.cpu_data()[i] * dimension_, top[0]->cpu_diff()[0], bottom[i]->mutable_gpu_diff());
    // }
    
    // //update Omega
    // caffe_gpu_set(num_of_tasks_ * num_of_tasks_, Dtype(0), Omega_.mutable_gpu_diff());
    // Dtype* A = Omega_.mutable_cpu_diff();
    // const Dtype* kernel = kernel_.cpu_data();
    
    // for(int i = 0;i < num_of_tasks_;++i){
    //     for(int j = 0;j < num_of_tasks_;++j){
    //         for(int k = 0;k < D_.cpu_data()[0];++k){
    //             int offset = (i * D_.cpu_data()[0] + k) * total_W_num_ + j * D_.cpu_data()[0] + k;
    //             A[i * num_of_tasks_ + j] += kernel[offset];
    //         }
    //     }
    // }
    
    // if(debug_info_){
    //     LOG(INFO) << "-------------------------------------begin";
    //     print_gpu_matrix(A, num_of_tasks_, num_of_tasks_, num_of_tasks_, num_of_tasks_);
    // }
    
    // caffe_cpu_matrix_sqrt(num_of_tasks_, A);

    // if(debug_info_){
    //     LOG(INFO) << "-------------------------------------aftar sqrt";
    //     print_gpu_matrix(A, num_of_tasks_, num_of_tasks_, num_of_tasks_, num_of_tasks_);    
    // }
    
    // //calculate trace
    // Dtype trace = 0;
    // for(int i = 0;i < num_of_tasks_;++i){
    //     trace += A[i * (num_of_tasks_ + 1)];
    // }
    // //divide by trace
    // caffe_scal(num_of_tasks_ * num_of_tasks_, 1 / trace, A);
    
    // if(debug_info_){
    //     LOG(INFO) << "-------------------------------------aftar divide trace";
    //     print_gpu_matrix(A, num_of_tasks_, num_of_tasks_, num_of_tasks_, num_of_tasks_);
    // }
    // //inverse
    // caffe_cpu_inverse(num_of_tasks_, A);
    
    // if(debug_info_){
    //     LOG(INFO) << "-------------------------------------aftar inverse";
    //     print_gpu_matrix(A, num_of_tasks_, num_of_tasks_, num_of_tasks_, num_of_tasks_);    
    // }
    // //copy to Omega
    // caffe_gpu_memcpy(sizeof(Dtype) * num_of_tasks_ * num_of_tasks_, A, Omega_.mutable_gpu_data());
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiTaskWeightLossLayer);

}  // namespace caffe
