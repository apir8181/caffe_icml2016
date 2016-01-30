#ifndef CAFFE_MULTI_TASK_WEIGHT_LOSS_LAYER_HPP_
#define CAFFE_MULTI_TASK_WEIGHT_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class MultiTaskWeightLossLayer : public LossLayer<Dtype> {
 public:
  explicit MultiTaskWeightLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiTaskWeightLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
													 const vector<Blob<Dtype>*>& top) {
		NOT_IMPLEMENTED;
	}
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		NOT_IMPLEMENTED;
	}
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  void Forward_gpu_FillWByTask(const vector<Blob<Dtype>*>& bottom);
  void Forward_gpu_FillAByTask();
  void Backward_gpu_UpdateOmega();
  void Backward_gpu_FillWByClass(const vector<Blob<Dtype>*>& bottom);
  void Backward_gpu_FillAByClass();
  void Backward_gpu_UpdateTheta();
  void Backward_gpu_Backprop(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

	bool debug_info_, debug_detail_;
  bool fronzen_omega_, fronzen_theta_;
  int num_task_, num_class_, num_feature_;
  vector<Blob<Dtype>*> W_by_task_; // W: C x D
  vector<Blob<Dtype>*> W_by_class_; // W: T X D
  Blob<Dtype> A_by_task_; // T x T
  Blob<Dtype> A_by_class_; // C x C
  Blob<Dtype> Omega_; // T x T
  Blob<Dtype> Theta_; // C x C
  Blob<Dtype> temp_D_C_ temp_D_D_, temp_D_T_;
};

}  // namespace caffe

#endif  
