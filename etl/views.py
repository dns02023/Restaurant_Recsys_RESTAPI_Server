from django.shortcuts import render
import os
import numpy as np
import pandas as pd
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
import pymysql.cursors
from recsys_restapi.settings import TRAIN_HOST, TRAIN_USER, TRAIN_PASSWORD, TRAIN_DB
from io import BytesIO, StringIO
import boto3
from recsys_restapi.settings import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_STORAGE_BUCKET_NAME

# Create your views here.
# POST: 사용자가 리뷰를 남기면 => train DB에 적재하는 과정
class ReviewETL(views.APIView):
    def post(self, request):
        request.POST._mutable = True
        user_name = request.data.pop('user_name')
        place_name = request.data.pop('place_name')
        rating = request.data.pop('rating')

        conn = pymysql.connect(host=TRAIN_HOST,
                                user=TRAIN_USER,
                                password=TRAIN_PASSWORD,
                                db=TRAIN_DB,
                                charset='utf8mb4')

        curs = conn.cursor()

        sql = "insert into train_reviews (user_name, place_name, rating) values (%s,%s,%s)"
        curs.execute(sql, (user_name[0], place_name[0], int(rating[0])))
        conn.commit()

        return Response(status=status.HTTP_200_OK)

# POST: train DB의 데이터들을 학습을 위해 전처리하여 csv 객체로 S3에 적재
class TrainETL(views.APIView):
     def post(self, request):
         request.POST._mutable = True
         csv_name = request.data.pop("csv_name")

         conn = pymysql.connect(host=TRAIN_HOST,
                                user=TRAIN_USER,
                                password=TRAIN_PASSWORD,
                                db=TRAIN_DB,
                                charset='utf8mb4',
                                autocommit=True,
                                cursorclass=pymysql.cursors.DictCursor)

         curs = conn.cursor()
         sql = "select * from train_reviews"
         curs.execute(sql)
         result = curs.fetchall()
         df = pd.DataFrame(result)

         users = df['user_name'].unique()
         places = df['place_name'].unique()
         user_indices = list()
         place_indices = list()

         for i in range(len(users)):
             temp = [i + 1, users[i]]
             user_indices.append(temp)

         for i in range(len(places)):
             temp = [i + 1, places[i]]
             place_indices.append(temp)

         user_df = pd.DataFrame(user_indices, columns=['user_id', 'user_name'])
         place_df = pd.DataFrame(place_indices, columns=['place_id', 'place_name'])

         df1 = pd.merge(df, user_df, on='user_name')
         final_df = pd.merge(df1, place_df, on='place_name')

         bucket = AWS_STORAGE_BUCKET_NAME
         csv_buffer = StringIO()

         final_df.to_csv(csv_buffer)
         s3_resource = boto3.resource('s3',
                                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
         s3_resource.Object(bucket, csv_name).put(Body=csv_buffer.getvalue())

         #train 하려면 csv 저장 전에 train용 user_id, place_id columns추가해야함.
         return Response(status=status.HTTP_200_OK)




