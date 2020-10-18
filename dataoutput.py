import sqlalchemy
import tensorflow as tf

username = 'postgres'  # DB username
password = 'COVID_type8eat'  # DB password
host = '34.86.177.25'  # Public IP address for your instance
port = '5432'
database = 'postgres'  # Name of database ('postgres' by default)

db_url = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(
    username, password, host, port, database)

engine = sqlalchemy.create_engine(db_url)

print("Connecting")

conn = engine.connect()
print("Connected")

testquery = "select tweet from twittertweet;"
result = conn.execute(testquery)
result_as_list = result.fetchall()
for row in result_as_list:
    print(row)

model = tf.keras.models.load_model('./model/gthack_model.h5')
predictions = model.predict(result_as_list)
print(predictions)

conn.close()
print("Connection Closed")