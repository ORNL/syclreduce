CXX = clang++
CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_75

# Alternative to cmake for quick testing.
test: example/example.cpp
	$(CXX) $(CXXFLAGS) -I include -o $@ $^
