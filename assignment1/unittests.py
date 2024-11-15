################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
import unittest
import numpy as np
import torch
from modules import LinearModule, SoftMaxModule, CrossEntropyModule
from modules import ELUModule


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    fx = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval

        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


class TestLosses(unittest.TestCase):

    def test_crossentropy_loss(self):
        np.random.seed(42)
        rel_error_max = 1e-5

        for test_num in range(10):
            N = np.random.choice(range(1, 100))
            C = np.random.choice(range(1, 10))
            y = np.random.randint(C, size=(N,))
            X = np.random.uniform(low=1e-2, high=1.0, size=(N, C))
            X /= X.sum(axis=1, keepdims=True)

            forward_our = CrossEntropyModule().forward(X, y)
            grads = CrossEntropyModule().backward(X, y)

            f = lambda _: CrossEntropyModule().forward(X, y)
            grads_num = eval_numerical_gradient(f, X, verbose=False, h=1e-5)
            
            self.assertLess(rel_error(grads_num, grads), rel_error_max)
    
    # def test_custom_loss(self):
    #     # N = np.random.choice(range(1, 100))
    #     # C = np.random.choice(range(1, 10))
    #     N = 2
    #     C = 3
    #     y = np.arange(N)
    #     # y = np.random.randint(C, size=(N,))
    #     X = np.random.uniform(low=1e-2, high=1.0, size=(N, C))
    #     X /= X.sum(axis=1, keepdims=True)

    #     X = np.array([[1.4, 0.4, 1.1, 0.1, 2.3]])
    #     y = np.array([0])

    #     X_softmax = torch.nn.functional.softmax(torch.from_numpy(X), dim=1).numpy()
    #     forward_our = CrossEntropyModule().forward(X_softmax, y)
    #     grads_our = CrossEntropyModule().backward(X_softmax, y)

    #     # Compare to pytorch
    #     pytorch_loss = torch.nn.CrossEntropyLoss(reduction='mean')
    #     X_torch = torch.from_numpy(X)
    #     y_torch = torch.from_numpy(y)
    #     X_torch.requires_grad = True
    #     y_torch.requires_grad = False
    #     forward_pytorch = pytorch_loss(X_torch, y_torch)
    #     grads_pytorch = torch.autograd.grad(forward_pytorch, X_torch)[0].numpy()

    #     forward_pytorch = forward_pytorch.detach().numpy()

    #     rel_error_max = 1e-5
    #     print(f'forward_our: {forward_our}')
    #     print(f'forward_pytorch: {forward_pytorch}')
    #     print(f'rel_error: {rel_error(forward_our, forward_pytorch)}')
    #     self.assertLess(rel_error(forward_our, forward_pytorch), rel_error_max)

    #     print(f'grads_our: {grads_our}')
    #     print(f'grads_pytorch: {grads_pytorch}')
    #     print(f'rel_error: {rel_error(grads_our, grads_pytorch)}')
    #     self.assertLess(rel_error(grads_our, grads_pytorch), rel_error_max)


class TestLayers(unittest.TestCase):

    def test_linear_backward(self):
        np.random.seed(42)
        rel_error_max = 1e-5

        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            D = np.random.choice(range(1, 100))
            C = np.random.choice(range(1, 10))
            x = np.random.randn(N, D)
            dout = np.random.randn(N, C)

            layer = LinearModule(D, C)

            _ = layer.forward(x)
            dx = layer.backward(dout)
            dw = layer.grads['weight']
            dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)
            dw_num = eval_numerical_gradient_array(lambda w: layer.forward(x), layer.params['weight'], dout)

            self.assertLess(rel_error(dx, dx_num), rel_error_max)
            self.assertLess(rel_error(dw, dw_num), rel_error_max)

    def test_elu_backward(self):
        np.random.seed(42)
        rel_error_max = 1e-6

        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            D = np.random.choice(range(1, 100))
            x = np.random.randn(N, D)
            dout = np.random.randn(*x.shape)

            layer = ELUModule(alpha=0.5)

            out = layer.forward(x)
            dx = layer.backward(dout)
            dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)

            self.assertLess(rel_error(dx, dx_num), rel_error_max)

    def test_softmax_backward(self):
        np.random.seed(42)
        rel_error_max = 1e-5

        for test_num in range(10):
            N = np.random.choice(range(1, 20))
            D = np.random.choice(range(1, 100))
            x = np.random.randn(N, D)
            dout = np.random.randn(*x.shape)


            layer = SoftMaxModule()

            _ = layer.forward(x)
            dx = layer.backward(dout)
            dx_num = eval_numerical_gradient_array(lambda xx: layer.forward(xx), x, dout)

            # print(f'I AM HERE dx shape: {dx.shape}')
            # print(f'I AM HERE dx_num shape: {dx_num.shape}')

            # print(f'dx: {dx}')
            # print(f'dx_num: {dx_num}')

            self.assertLess(rel_error(dx, dx_num), rel_error_max)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLosses)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestLayers)
    unittest.TextTestRunner(verbosity=2).run(suite)
