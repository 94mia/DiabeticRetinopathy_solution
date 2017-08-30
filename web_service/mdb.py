import pymysql

__all__ = ['start_db_conn', 'close_db_conn', ]

cmd_showdb = 'show databases'
cmd_createdb = 'create database if not exists dr_images_db'
cmd_usedb = 'use dr_images_db'
cmd_showtb = 'show tables'
cmd_createtb = """create table if not exists dr_image_tb (
    id char(20) not null,
    imagepath char(200) not null,
    algolevel int,
    doctorlevel int
)"""

def start_db_conn():
    # connect db
    db = pymysql.connect('localhost', 'root')
    cursor = db.cursor()
    # create db
    cursor.execute(cmd_createdb)
    cursor.execute(cmd_showdb)
    data = cursor.fetchall()
    print(data)
    # create table
    cursor.execute(cmd_usedb)
    cursor.execute(cmd_createtb)
    cursor.execute(cmd_showtb)
    print(cursor.fetchall())

    return db, cursor

def close_db_conn(db):
    db.close()

start_db_conn()

