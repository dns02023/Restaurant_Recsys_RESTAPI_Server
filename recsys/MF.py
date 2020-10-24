import numpy as np

class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        R: 초기 평점 matrix
        k: 학습할 latent parameter 갯수
        """
        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose


    def fit(self):
        """
        * Matrix Factorization 으로 latent factor weight와 bias를 학습
        * _b (global bias): input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        """

        # latent feature 초기화
        # 정규분포 내에서 랜덤 초기화
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # bias 초기화
        # user bias, item bias: 0으로 초기화
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        # global bias: 전체 평균으로 초기화(존재하는 평가에 대해서)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):

            # rating이 존재하는 index를 기준으로 training
            # 전체 샘플(R의 모든 값)을 싹 다 훑음
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d, cost = %.4f" % (epoch + 1, cost))


    def cost(self):
        """
        MSE return
        전체 cost 값구하기
        """

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        xi, yi = self._R.nonzero()
        #0이 아닌 값의 index를 반환 함.
        predicted = self.reconstruct()
        cost = 0

        # R에서의 nonzero 값들이 바로 training 데이터인 개념
        # => (실제값 - 예측값)^2을 각 평가(i,j)마다 수행
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - predicted[x, y], 2)
        return cost/len(xi)


    def gradient(self, error, i, j):
        """
        latent factor에 대한 gradient 계산
        error: rating - prediction (실제값 - 예측값)
        i: user index (i번째 유저)
        j: item index (j번째 아이템)
        """
        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        #error앞의 2는 생략
        return dp, dq
    # i번째 유저와 j번째 아이템의 평점 하나에 대한 gradient


    def gradient_descent(self, i, j, rating):
        """
        i: user latent factor matrix의 i번째 유저에 대해서
        j: item latent factor matrix의 j번째 아이템에 대해서
        경사하강 수행
        rating: rating matrix에서 (i,j)번째 값 => 실제 값 (i번째 유저가 j번째 아이템에 내린 실제 평점)
        """

        prediction = self.get_prediction(i, j)
        error = rating - prediction

        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq


    def get_prediction(self, i, j):
        """
        i번째 유저가 j번째 아이템에 내릴 평점을 예측
        """
        #P, Q의 내적으로 비어있던 RATING값을 예측하고 global bias와 p, q 각각의 bias값들을 더해 준다.
        #_b_P[i]: i번째 user의 bias
        #_b_Q[j]: j번쨰 item의 bias
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)



    def reconstruct(self):
        """
        학습 결과물인 reconstructed rating matrix
        """

        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)


