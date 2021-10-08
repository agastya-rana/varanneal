from autograd import grad
import autograd.numpy as np

class rand():
    def s(self, x):
        y = np.exp(-2.0 * x)
        return (1.0 - y) / (1.0 + y)

    def g(self, x):
        gr = grad(self.s)
        return gr(x)

a = rand()
print(a.g(2.0))
