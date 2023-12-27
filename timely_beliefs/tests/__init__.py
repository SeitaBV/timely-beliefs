from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql://flexmeasures_test:flexmeasures_test@127.0.0.1/flexmeasures_test")
Session = sessionmaker(bind=engine)
session = Session()
