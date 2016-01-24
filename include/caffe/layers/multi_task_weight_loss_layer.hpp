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
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		NOT_IMPLEMENTED;
	}
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
	bool debug_info_, debug_detail_;
  int num_tasks_, num_classes_, feature_dim_;
  Blob<Dtype> data_;
	Blob<int> task_start_index_, task_end_index_;
	Blob<int> data2task_;
	Blob<Dtype> pairwise_sqr_distance_;
	Blob<Dtype> pairwise_kernel_;
	Blob<Dtype> loss_;
  Blob<Dtype> A_;
	Dtype sigma_;
  
};

}  // namespace caffe

#endif  
