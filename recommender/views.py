import os
import numpy as np
import pandas as pd
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from recsys import MF
from io import BytesIO
import boto3
from recsys_restapi.settings import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_STORAGE_BUCKET_NAME
import heapq

s3_resource = boto3.resource('s3',
                                     aws_access_key_id=AWS_ACCESS_KEY_ID,
                                     aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

s3_client = boto3.client('s3',
                                 aws_access_key_id=AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                                 )


# Create your views here.
class Similarity(views.APIView):
    def post(self, request):

        places = pd.read_csv('real_place_data.csv')
        places['place_tags'] = places['place_tags'].str.split(pat="#")
        places['place_tags'] = places['place_tags'].apply(lambda x: " ".join(x))

        count = CountVectorizer()
        count_matrix = count.fit_transform(places['place_tags'])

        cosine_sim = cosine_similarity(count_matrix, count_matrix)

        bucket = AWS_STORAGE_BUCKET_NAME
        npy_buffer = BytesIO()

        np.save(npy_buffer, cosine_sim)
        s3_resource.Object(bucket, 'cosine_similarity.npy').put(Body=npy_buffer.getvalue())

        return Response(status=status.HTTP_200_OK)

class CBRecommend(views.APIView):
    def get(self, request, id):
        obj = s3_client.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key='cosine_similarity.npy')
        cosine_sim = np.load(BytesIO(obj['Body'].read()))

        places = pd.read_csv("real_place_data.csv")
        idx = int(id) - 1

        sim_scores = list(enumerate(cosine_sim[idx]))
        # 유사도 순으로 정렬
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:6]

        place_indices = [i[0] for i in sim_scores]
        result_place = places.iloc[place_indices].copy()
        rec_ids = list(result_place['id'])

        return Response(rec_ids, status=status.HTTP_200_OK)

class Train(views.APIView):
    def post(self, request):
        request.POST._mutable = True
        data_name = request.data.pop("data_name")
        model_name = request.data.pop("model_name")

        obj = s3_client.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=data_name)
        rating = pd.read_csv(BytesIO(obj['Body'].read()))

        # 데이터들을 수집할때마다 update해주고, train

        ratings_df = rating.groupby(['user_id', 'place_id'], as_index=False).mean()
        # 한 유저가 동일 식당을 여러번 평가하는 경우 하나로 압축(평균)

        ratings = ratings_df.pivot(
            index='user_id',
            columns='place_id',
            values='rating'
        ).fillna(0)

        matrix = ratings.values

        mf_model = MF.MatrixFactorization(matrix, k=10, learning_rate=0.05, reg_param=0.01, epochs=10, val_prop=0.2, tolerance=3)
        mf_model.fit()
        pred_matrix = mf_model.reconstruct()

        bucket = AWS_STORAGE_BUCKET_NAME
        npy_buffer = BytesIO()

        np.save(npy_buffer, pred_matrix)
        s3_resource.Object(bucket, model_name).put(Body=npy_buffer.getvalue())

        return Response(status=status.HTTP_200_OK)

# GET이 아니라 POST 인 이유? => 유저 닉네임을 인자로 받아야 하는데, GET을 사용하면, 보안상 안 좋을 것 같음
class CFRecommend(views.APIView):
    def post(self, request):
        request.POST._mutable = True

        data_obj = s3_client.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key='0727_train_rating_data.csv')
        model_obj = s3_client.get_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key='0730_pred_matrix.npy')

        rating = pd.read_csv(BytesIO(data_obj['Body'].read()))
        pred_matrix = np.load(BytesIO(model_obj['Body'].read()))

        #한번 학습한 결과를 바탕으로 추천
        buffer = request.data.pop('sofo_name')
        target = buffer[0]

        temp = rating[rating['user_name'] == target].user_id.unique()
        if len(temp) == 0:
            return Response('No Review', status=status.HTTP_200_OK)

        # 이미 사용자가 평가를 내린 맛집은 추천 안 함
        remove = rating[rating['user_id'] == temp[0]].place_id.unique()
        target_pred_rating = pred_matrix[temp[0] - 1]

        recommend_list = list()
        for i in range(len(target_pred_rating)):
            if (i + 1) in remove:
                continue
            else:
                if len(recommend_list) < 6:
                    heapq.heappush(recommend_list, (target_pred_rating[i], i + 1))
                else:
                    if recommend_list[0][0] < target_pred_rating[i]:
                        heapq.heappop(recommend_list)
                        heapq.heappush(recommend_list, (target_pred_rating[i], i + 1))

        response_list = [recommend_list[i][1] for i in range(len(recommend_list))]
        return Response(response_list, status=status.HTTP_200_OK)









        
















