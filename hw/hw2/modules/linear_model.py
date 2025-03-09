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
        self.obj_loss = loss_function
        self.batch_size = batch_size
        self.alpha = step_alpha
        self.beta = step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.seed = random_seed
        self.model_answer = None

    # TODO: валидация
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
            w = np.array([1.0 for i in range(X.shape[1])])
        else:
            w = np.copy(w_0)
        i = 1
        if self.batch_size is not None:
            iters_per_epoches = X.shape[0] // self.batch_size
            rand_gen = np.random.default_rng(self.seed)
            ind = np.array([i for i in range(X.shape[0])])
        else:
            iters_per_epoches = 1
            rand_gen = None
        w_prev = None
        epoche_count = 0
        if trace:
            history = {
                'time': [],
                "func": [],
                "func_val": []
            }
        else:
            history = None
        while i <= self.max_iter and (w_prev is None or np.sum(np.pow(w_prev - w, 2)) < self.tol):
            if iters_per_epoches != 1:
                # Попробуем делать честное сэмплирование
                data_ind = rand_gen.choice(ind, self.batch_size, replace=False)
                data = X[data_ind]
                y_data = y[data_ind]
            else:
                # искренне надеюсь, что здесь
                # просто перевесятся указатели
                data = X
                y_data = y
            grad = self.obj_loss.grad(data, y_data, w)
            w_prev = np.copy(w)
            w = w - (self.alpha / pow(i, self.beta)) * grad
            if trace and epoche_count == 0 or i // (epoche_count * iters_per_epoches) > 0:
                epoche_count += 1
                # Создаём новую запись об эпохе
                # TODO
            # Записываем дамп
            # TODO
            if trace:
                pass
            i += 1

        self.model_answer = w
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
        pass

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d model weights vector.
        """
        pass

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
        pass
