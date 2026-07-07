import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    os.environ.get("TB_TEST_DB_URL", "postgresql://tbtest:tbtest@127.0.0.1/tbtest")
)
Session = sessionmaker(bind=engine)
session = Session()
