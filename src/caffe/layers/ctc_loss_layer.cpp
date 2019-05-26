//// ctc转录层 参数
/* // proto
message CtcLossParameter {
    optional uint32 alphabet_size = 1 [default = 0]; // 字符数量(每个特征预测的字符类别数)
    optional uint32 time_step = 3 [default = 0];     // 时间步长  lstm层输出通道数量
    optional int32 blank_label = 4 [default = 0];    // 空白字符标签
}
message ContinuationIndicatorParameter {
    optional uint32 time_step = 1 [default = 0];
    optional uint32 batch_size = 2 [default = 0];
} 
*/ 

#include <caffe/layers/ctc_loss_layer.hpp>
namespace caffe {

template <typename Dtype>
void CtcLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,  // 层输入
    const vector<Blob<Dtype>*>& top)     // 层输出
{
  LossLayer<Dtype>::LayerSetUp(bottom, top);// 损失层初始化
    
  CtcLossParameter param = this->layer_param_.ctc_loss_param();
  blank_label_ = param.blank_label();      // 空白字符
  alphabet_size_ = param.alphabet_size();  // 字符数量
  CHECK_GT(alphabet_size_, 0) << "The size of alphabeta should be greater than 0.";
  int mini_batch = bottom[0]->shape()[1];  // 输入通道数量 
  //int label_length = param.label_length();
  //CHECK_GT(label_length, 0) << "The length of label sequence should be greater than 0.";
  label_lengths_ = vector<int>(mini_batch);// 需要识别的字符总长度(输入特征通道数量) ？？
  int input_length = param.time_step();    // 输入长度？？
  CHECK_GT(input_length, 0) << "The time step should be greater than 0.";
  input_lengths_ = vector<int>(mini_batch, input_length);
}

template <typename Dtype>
void CtcLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, // 层输入
    const vector<Blob<Dtype>*>& top)    // 层输出
{
    LossLayer<Dtype>::Reshape(bottom, top);
    // bottom[0]: TxNxC  C: alphabeta_size
    // bottom[1]: NxL    L:length of label swq
    CHECK_EQ(bottom[1]->shape()[0], bottom[0]->shape()[1])
        << "The input blobs should have same dimensions.";
}

template <>
void CtcLossLayer<double>::Forward_cpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void CtcLossLayer<Dtype>::FlattenLabels(const Blob<Dtype>* label_blob)
{
    const Dtype* label_data = label_blob->cpu_data();
    CHECK_EQ(label_lengths_.size(), label_blob->shape()[0]) << "different dimensions!";
    total_label_length_ = 0;
    flat_labels_.clear();
    for(int b = 0; b < label_blob->shape()[0]; ++b)
    {
      label_lengths_[b] = 0;
      for(int c = 0; c < label_blob->shape()[1]; ++c)
      {
        int label = static_cast<int>(*label_data++);
        CHECK_GE(label, 0) << "label should be greater than or equal with 0.";
        CHECK_LT(label, alphabet_size_) << "label should be less than alphabet size.";
        if(label != blank_label_)       // 非空白字符
        {
          ++label_lengths_[b];
          ++total_label_length_;
          flat_labels_.push_back(label);// 保存所有非空白字符
        }
      }
    }
    CHECK_GT(total_label_length_, 0) << "total length should be greater than 0.";
}

template <typename Dtype>
void CtcLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, // 层输入
    const vector<Blob<Dtype>*>& top)    // 层输出
{
    auto options = ctcOptions{};
    options.loc = CTC_CPU;
    options.num_threads = 1;
    options.blank_label = blank_label_;
    
    int mini_batch = bottom[0]->shape(1);
    int alphabet_size = alphabet_size_;

    const Dtype* const activations = bottom[0]->cpu_data();// 输入
    Dtype* gradients = bottom[0]->mutable_cpu_diff();
    Dtype* cost = new Dtype[mini_batch];
    
    FlattenLabels(bottom[1]);
    
    size_t size_bytes;
    CHECK_CTC_STATUS(get_workspace_size(label_lengths_.data(),
                    input_lengths_.data(), alphabet_size,
                    mini_batch, options, &size_bytes));
    char* workspace = new char[size_bytes];

    CHECK_CTC_STATUS(compute_ctc_loss(activations, gradients,
                     flat_labels_.data(),
                     label_lengths_.data(), input_lengths_.data(),
                     alphabet_size, mini_batch, cost,
                     workspace, options));
    
    // ===========================================================
    Dtype loss = std::accumulate(cost, cost + mini_batch, Dtype(0));
    
    top[0]->mutable_cpu_data()[0] = loss / mini_batch;
    
    delete[] cost;
    delete[] workspace;
}

template <>
void CtcLossLayer<double>::Backward_cpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<double>*>& bottom)
{
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void CtcLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if(propagate_down[0]) {
        caffe_scal(bottom[0]->count(), top[0]->cpu_diff()[0],
                   bottom[0]->mutable_cpu_diff());
    }
}

#ifdef CPU_ONLY
STUB_GPU(CtcLossLayer);
#endif

INSTANTIATE_CLASS(CtcLossLayer);
REGISTER_LAYER_CLASS(CtcLoss);

}  // namespace caffe
