import numpy as np
import time


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0, 
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.alpha = step_alpha
        self.beta = step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.seed = random_seed
        self._coef = None

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector for initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        if w_0 is None:
            w = np.zeros(X.shape[1])
            w[0] = 1
        else:
            w = np.copy(w_0)
        i = 1
        if self.batch_size is not None:
            iters_per_epoch = X.shape[0] // self.batch_size
            rand_gen = np.random.default_rng(self.seed)
        else:
            iters_per_epoch = 1
            rand_gen = None
        w_prev = None
        epoche_count = 1
        if trace:
            history = {
                'time': [0],
                "func": [],
                "func_val": [],
                "log": []
            }
        while i <= self.max_iter and (w_prev is None or np.sqrt(np.sum(np.pow(w_prev - w, 2))) > self.tol):
            # history['log'].append(i)
            if trace:
                start_time = time.time()
            
            if self.batch_size is not None and i % iters_per_epoch == 1:  
                eta = self.alpha / np.float_power(epoche_count, self.beta)  
                perm = rand_gen.permutation(X.shape[0])
            elif self.batch_size is None:
                eta = self.alpha / np.float_power(i, self.beta)
            
            if self.batch_size is not None:
                batch_start = ((i - 1) % iters_per_epoch) * self.batch_size
                batch_end = min(batch_start + self.batch_size, X.shape[0])
                data_ind = perm[batch_start:batch_end]
                data = X[data_ind]
                y_data = y[data_ind]
                if trace:
                    history['log'].append([batch_start, batch_end])
            else:
                # искренне надеюсь, что здесь
                # просто перевесятся указатели
                data = X
                y_data = y
            # eta = self.alpha / np.float_power(i, self.beta)
            grad = self.loss_function.grad(data, y_data, w)
            w_prev = np.copy(w)
            w = w - eta * grad
            # Временная отсечка ставится, как только перестали считать
            # обязательную программу
            if trace:
                end_time = time.time()
                history['log'].append(f"delta_w = {np.sqrt(np.sum(np.pow(w_prev - w, 2)))}")
                # Запись времени производится на каждой итерации
                history['time'][epoche_count - 1] += (end_time - start_time)
            if i % iters_per_epoch == 0:
                epoche_count += 1
                if trace:
                    history['time'].append(0)
                    history['func'].append(self.loss_function.func(data, y_data, w))
                    if X_val is not None and y_val is not None:
                        history['func_val'].append(self.loss_function.func(X_val, y_val, w))
                    else:
                        history['func_val'].append(0)
            

            i += 1

        self._coef = w
        if trace:
            return history

            
        


        

    def predict(self, X):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        return X @ self._coef

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d model weights vector.
        """
        return self._coef

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function(X, y, self._coef)
