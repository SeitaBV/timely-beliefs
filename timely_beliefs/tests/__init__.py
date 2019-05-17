from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from timely_beliefs.tests.examples import df_example

example_df = df_example()

engine = create_engine("postgresql://tbtest:tbtest@127.0.0.1/tbtest")
Session = sessionmaker(bind=engine)
session = Session()
