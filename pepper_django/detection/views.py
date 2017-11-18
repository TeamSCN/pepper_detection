# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse
from object_detection.pepper_detection import django #import django/pepper/detection/object_detection/pepper_detection/django.py
from PIL import Image
import datetime
import pymysql
image=Image.open('/home/cc/pepper/django/pepper/detection/image2.jpg')#test example
image_np=django.load_image_into_numpy_array(image)#change picture to numpy array in order to recognition


def mysql_add_att(conn, cursor, usr_id, add_date, add_time,
                  att_sta):  # usr_id: Student id, add_date: Date, add_time: Time, att_sta: attendance status
    if att_sta == True:
        att_sta_temp = 1
    else:
        att_sta_temp = 0
    sqlstr = "INSERT INTO attmgt_attendance(student_id, date, time, att) VALUES(%d, '%s', '%s', %d)" % (
    usr_id, str(add_date), str(add_time), att_sta_temp)

    try:
        cursor.execute(sqlstr)
        conn.commit()
        # print "add"
    except:
        conn.rollback()
        print 'failure'

    #pepper_date = datetime.datetime.now().strftime("%Y-%m-%d")
    #pepper_time = datetime.datetime.now().strftime("%H:00:00")

def index(request):
    db = pymysql.connect('192.168.1.244', 'root', 'haralabm1', 'Attendance', charset='utf8')
    cursor = db.cursor()
    #sql='insert into attmgt_attendance(date,att,student_id,time) values(%s,%s,%s,%s)'
    result= django.detect_object(image_np)#return true or false
    att=0
    if result ==True:
        att=1
    student_id=2
    date=datetime.datetime.now().date()
    time=datetime.datetime.now().time()
    mysql_add_att(db,cursor,student_id,date,time,result)
    db.close()

    return HttpResponse(result)

# Create your views here.
