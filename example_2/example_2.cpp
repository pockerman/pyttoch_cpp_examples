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
	Net(uint_t n, uint_t m);
	
	// forward
	torch::Tensor forward(torch::Tensor input);
	
private:

	torch::Tensor W;
	torch::Tensor b;

};

Net::Net(uint_t n, uint_t m)
:
W(),
b()
{
 W = register_parameter("W", torch::randn({n, m}));
 b = register_parameter("b", torch::randn(m));
}

torch::Tensor 
Net::forward(torch::Tensor input){
	return torch::addmm(b, input, W);
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

  
  
  Net net(4, 5);
  for (const auto& p : net.parameters()) {
    std::cout << p << std::endl;
  }
  
  
  
  return 0;
}
