from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql://tbtest:tbtest@127.0.0.1/tbtest")
Session = sessionmaker(bind=engine)
session = Session()
