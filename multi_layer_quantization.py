import numpy as np

np.random.seed(10)


NUM_FEATURES = 3
HIDDEN_FEATURES = 15
VAL_MIN = -10.
VAL_MAX = 10.
BATCH_INPUT = 5
BATCH_CALIB = 10000


def apply_network(network, x):
    y = x
    for layer in network:
        y = layer(y)
    return y
    
    
def compute_scale(x, n_bits=8):
    if isinstance(x, np.ndarray):
        x = [x]
    max_val = max([np.max(v) for v in x])
    min_val = min([np.min(v) for v in x])
    return (max_val - min_val) / (2**n_bits - 1)
   
   
def quantize(x, s, n_bits=8):
        signed_range = 2**(n_bits-1) - 1
        return np.clip(np.round(x / s ), -signed_range, signed_range).astype(np.int8)


class LinLayer:

    def __init__(self, n_in, n_out):
        self.W = np.random.uniform(size=(n_in, n_out), low=-1., high=1.)
        self.b = np.random.uniform(size=(n_out,), low=-0.1, high=0.1)

    def __call__(self, x):
        return  np.matmul(x, self.W) + self.b


class QuantizedLinLayer:

    def __init__(self, linear_layer, x_calib, s_calib):
        s_params = compute_scale([linear_layer.W, linear_layer.b])
        self.Wq = quantize(linear_layer.W, s_params)
        self.bq = quantize(linear_layer.b, s_params * s_calib)
        self.s = s_params * s_calib
        self.s_req = self._calibrate(x_calib)
        
    def _apply_linear(self, xq):
        return np.matmul(xq.astype(np.int32), self.Wq.astype(np.int32)) + self.bq.astype(np.int32)
    
    def __call__(self, xq):
        yq = self._apply_linear(xq)
        return quantize(yq, self.s_req) # requantize
                
    def _calibrate(self, x_calib):
        y = self._apply_linear(x_calib)
        return compute_scale(y)
    
    @property
    def output_scale(self):
        return self.s * self.s_req


def quantize_network(network, calib_data):
    quantized_network = []
    s_calib = s = compute_scale(calib_data)
    x_calib = quantize(calib_data, s_calib)
    
    for layer in network:
        quantized_layer = QuantizedLinLayer(layer, x_calib, s_calib)
        quantized_network.append(quantized_layer)
        x_calib = quantized_layer(x_calib)
        s_calib = quantized_layer.output_scale
    return quantized_network


x = np.random.uniform(size=(BATCH_INPUT, NUM_FEATURES), low=VAL_MIN, high=VAL_MAX)
network = [LinLayer(NUM_FEATURES, HIDDEN_FEATURES),
           LinLayer(HIDDEN_FEATURES, HIDDEN_FEATURES),
           LinLayer(HIDDEN_FEATURES, NUM_FEATURES)]
print("input: ", x)

# regular forward pass
y = apply_network(network, x)
print("\ntarget output: ", y)

# large calibration set for statistical robustness
calib_data = np.random.uniform(size=(BATCH_CALIB, NUM_FEATURES), low=VAL_MIN, high=VAL_MAX)
quantized_network = quantize_network(network, calib_data)
final_s = quantized_network[-1].output_scale

# compute quantized network output and dequantize
xs = compute_scale(x)
xq = quantize(x, xs)
yq = apply_network(quantized_network, xq)
y_deq = quantized_network[-1].output_scale * yq.astype(np.float32)
print("\ndequantized output: ", y_deq)
print("\nquantization errors: ", np.abs(y - y_deq))
