from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# This engine and session should move to tests, only useful there. Base is needed by packages using tb,
# but we might put it in the index.
engine = create_engine("postgresql://tbtest:tbtest@127.0.0.1/tbtest")
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()
