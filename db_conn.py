DB_HOST = "localhost"
DB_NAME = "circopt"
DB_USER = "ioanamoflic"
DB_PASS = ""

import psycopg2
import logging

logging.basicConfig(filename='logfile.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')


def get_qt():
    entries = []
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS)
        cursor = conn.cursor()

        cursor.execute("select * from qtable")
        entries = cursor.fetchall()

    except (Exception, psycopg2.Error) as error:
        logging.error(error)

    finally:
        if conn:
            cursor.close()
            conn.close()

    return entries


