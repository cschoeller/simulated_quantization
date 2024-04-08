import numpy as np

np.random.seed(42)


x = np.random.uniform(size=(3,3))
w = np.random.uniform(size=(3,3))
b = np.random.uniform(size=(3,1))

print("Float tensors:")
print("x: ", x)
print("w: ", w)
print("b: ", b)

y = np.matmul(w, x) + b
print("y: ", y)



def quantize(x, max_val=255):
    return np.clip(np.round(x * max_val), 0, max_val).astype(np.int32)

xq = quantize(x)
wq = quantize(w)
bq = quantize(b, max_val=255**2)

print("")
print("Quantizied tensors:")
print("xq", xq)
print("wq", wq)
print("bq", bq)



yq = np.matmul(wq, xq) + bq
print("yq: ", yq)

scale = 1./255**2
y_deq = scale * yq
print("y_deq: ", y_deq)


q_errors = np.abs(y_deq - y)
print("")
print("quantization errors: ", q_errors)