"""
참고 논문: 'Matrix factorization techniques for recommender systems', Yehuda Koren, 2009
"""
import numpy as np
import copy


class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, val_prop, tolerance):
        """
        R: 초기 평점 matrix
        k: 학습할 latent parameter 갯수
        val_prop: validation set 비율
        tolerance: validation mse가 이전 단계보다 증가하는 횟수가 tolerance보다 커지면 early stop 한다.
        """
        self._R = R
        self._train_R = copy.deepcopy(self._R)
        self._val_prop = int(R.size * val_prop)
        self._val_user_indices = [np.random.choice(range(R.shape[0])) for _ in range(self._val_prop)]
        self._val_place_indices = [np.random.choice(range(R.shape[1])) for _ in range(self._val_prop)]
        self._train_R[self._val_user_indices, self._val_place_indices] = 0

        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._tolerance = tolerance

    def validate(self):
        """
        validate_MSE return
        split한 data에 대해서 validation error 구하기

        """
        pred_matrix = self.reconstruct()
        validate_mse = 0.0
        count = 0
        for i in range(len(self._val_user_indices)):
            if self._R.item(self._val_user_indices[i], self._val_place_indices[i]) != 0:
                count = count + 1
                validate_mse = validate_mse + (
                            self._R.item(self._val_user_indices[i], self._val_place_indices[i]) - pred_matrix.item(
                        self._val_user_indices[i], self._val_place_indices[i])) ** 2
        validate_mse = (validate_mse / count)
        return validate_mse

    def fit(self):
        """
        Matrix Factorization 으로 latent factor weight와 bias를 학습
        _b (global bias): input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
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
        self._b = np.mean(self._train_R[np.where(self._train_R != 0)])

        # train while epochs
        self._training_process = []

        overfit_count = 0
        for epoch in range(self._epochs):

            # rating이 존재하는 index를 기준으로 training
            # 전체 샘플(R의 모든 값)을 싹 다 훑음
            # 논문의 'Stochastic gradient descent' part 참고
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._train_R[i, j] > 0:
                        # 논문: 'loops through all ratings in the training set' 구현
                        self.gradient_descent(i, j, self._train_R[i, j])
            train_cost = self.cost()
            # 5 epoch 마다 validate
            if (epoch % 5) == 0:
                validation_cost = self.validate()
                if epoch > 0:
                    # 이전 5 epoch 보다 validation cost가 증가 했다면, overfit_count가 증가
                    if self._training_process[-1][2] < validation_cost:
                        overfit_count = overfit_count + 1
                        # overfit_count가 tolerance를 초과하게 되면 overfit 방지를 위해 early stop
                        if overfit_count > self._tolerance:
                            self._training_process.append([epoch, train_cost, validation_cost])
                            break

                self._training_process.append([epoch, train_cost, validation_cost])

        return self._training_process

    def cost(self):
        """
        MSE return
        전체 cost 값구하기
        """

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        xi, yi = self._train_R.nonzero()
        # 0이 아닌 값의 index를 반환 함.
        predicted = self.reconstruct()
        cost = 0

        # R에서의 nonzero 값들이 바로 training 데이터인 개념
        # => (실제값 - 예측값)^2을 각 평가(i,j)마다 수행
        for x, y in zip(xi, yi):
            cost += pow(self._train_R[x, y] - predicted[x, y], 2)
        return cost / len(xi)

    def gradient(self, error, i, j):
        """
        latent factor에 대한 gradient 계산
        error: rating - prediction (실제값 - 예측값)
        i: user index (i번째 유저)
        j: item index (j번째 아이템)
        """
        # 논문 수식 참고
        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        # error앞의 2는 생략
        return dp, dq

    # i번째 유저와 j번째 아이템의 평점 하나에 대한 gradient

    def gradient_descent(self, i, j, rating):
        """
        i: user latent factor matrix의 i번째 유저에 대해서
        j: item latent factor matrix의 j번째 아이템에 대해서
        경사하강 수행
        rating: rating matrix에서 (i,j)번째 값 => 실제 값 (i번째 유저가 j번째 아이템에 내린 실제 평점)
        """

        # 논문 'For each given training case, the system predicts rui and computes the associated prediction error' 구현
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
        # P, Q의 내적으로 비어있던 RATING값을 예측하고 global bias와 p, q 각각의 bias값들을 더해 준다.
        # _b_P[i]: i번째 user의 bias
        # _b_Q[j]: j번쨰 item의 bias
        # 논문 'bui = average rating over all items + bi + bu' 구현 => r^ui = bui + dot product (Pi, Qj)
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)

    def reconstruct(self):
        """
        학습 결과물인 reconstructed rating matrix
        """

        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)






