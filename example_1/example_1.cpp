#include <torch/torch.h>

#include <iostream>
#include <vector>

int main() {


  if(torch::cuda::is_available()){
  	std::cout<<"CUDA is available on this machine"<<std::endl;
  }
  else{
  	std::cout<<"CUDA is not available on this machine"<<std::endl;
  }

  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
  
  std::vector<double> data(3, 2.0);
  auto tensor_from_data_1 = torch::tensor(data);
  std::cout << tensor_from_data_1 << std::endl; 
  
  data[0] = data[1] = data[2] = 1.0;
  auto tensor_from_data_2 = torch::tensor(data);
  std::cout << tensor_from_data_2 << std::endl; 
  
  auto sum = tensor_from_data_2 + tensor_from_data_1;
  std::cout << sum << std::endl;
    
    
  if(torch::cuda::is_available()){
  
   // create a tensor and send it to the GPU
   auto cuda_tensor = torch::tensor({1.0, 2.0, 3.0}).to("cuda");
   
  }
  
  // compute element-wise product
  auto tensor1 = torch::tensor({1.0, 2.0, 3.0});
  auto product = tensor1 * tensor1;
  std::cout << product << std::endl;
  
  
  
  return 0;
}
