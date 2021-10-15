#include <torch/torch.h>

#include <iostream>
#include <vector>

namespace example
{

typedef std::size_t uint_t;

class Net: public torch::nn::Module
{

public:

	//
	Net(uint_t input_size, uint_t output_size);
	
	// forward
	torch::Tensor forward(torch::Tensor input);
	
private:

	torch::nn::Linear linear;
	torch::Tensor bias_;

};

Net::Net(uint_t input_size, uint_t output_size)
:
linear(register_module("linear", torch::nn::Linear(input_size, output_size))),
bias_()
{
	bias_ = register_parameter("b", torch::randn(output_size));
}

torch::Tensor 
Net::forward(torch::Tensor input){
	return linear(input) + bias_;
}

}

int main() {


  using namespace  example;
  
  if(torch::cuda::is_available()){
  	std::cout<<"CUDA is available on this machine"<<std::endl;
  }
  else{
  	std::cout<<"CUDA is not available on this machine"<<std::endl;
  }
  
  // create data
  std::vector<double> x_train(11, 0.0);
  std::vector<double> y_train(11, 0.0);
  
  for(uint_t i=0; i<x_train.size(); ++i){
  	x_train[i] = static_cast<double>(i);
  	y_train[i] = 2*static_cast<double>(i) + 1;
  }
                  
  auto x_tensor = torch::from_blob(x_train.data(), {int(y_train.size()), int(x_train.size()/y_train.size())});
  auto y_tensor = torch::from_blob(y_train.data(), {int(y_train.size()), 1});
  
  Net net(1, 1);
  for (const auto& p : net.parameters()) {
    std::cout << p << std::endl;
  }
  
  torch::nn::MSELoss mse;
  torch::optim::SGD sgd(net.parameters(), 0.01);
  
  for(uint_t e=0; e<100; ++e){
  
  	sgd.zero_grad();
  	
  	auto outputs = net.forward(x_tensor);
  	auto loss = mse(outputs, y_tensor);
  	
  	// get gradients w.r.t to parameters
    	loss.backward();

    	// update parameters
    	sgd.step();
    	std::cout<<"Epoch="<<e<<" loss="<<loss<<std::endl;
  }
  
  return 0;
}
