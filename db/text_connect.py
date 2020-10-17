import sqlalchemy

username = 'postgres'  # DB username
password = 'COVID_type8eat'  # DB password
host = '34.86.177.25'  # Public IP address for your instance
port = '5432'
database = 'postgres'  # Name of database ('postgres' by default)

db_url = 'postgresql+psycopg2://{}:{}@{}:{}/{}'.format(
    username, password, host, port, database)

engine = sqlalchemy.create_engine(db_url)

conn = engine.connect()