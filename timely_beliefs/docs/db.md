# Database storage

## Table of contents

1. [Derived database classes](#derived-database-classes)
1. [Table creation and session](#table-creation-and-session)
1. [Subclassing](#subclassing)
1. [Queries](#queries)

## Derived database classes

The timely-beliefs library supports persisting your beliefs data in a database.
All relevant classes have a subclass which also derives from [sqlalchemy's declarative base](https://docs.sqlalchemy.org/en/13/orm/extensions/declarative/index.html?highlight=declarative).

The timely-beliefs library comes with database-backed classes for the three main components of the data model - `DBTimedBelief`, `DBSensor` and `DBBeliefSource`.
Objects from these classes can be used just like their super classes, so for instance `DBTimedBelief` objects can be used for creating a `BeliefsDataFrame`.

### Table creation and storage

You can let sqlalchemy create the tables in your database session and start using the DB classes (or subclasses, see below) and program code without much work by yourself.
The database session is under your control - where or how you get it, depends on the context you're working in.
Here is an example how to set up a session and also have sqlachemy create the tables:

    from sqlalchemy.orm import sessionmaker
    from timely_beliefs.db_base import Base as TBBase

    SessionClass = sessionmaker()
    session = None

    def create_db_and_session():
        engine = create_engine("your-db-connection-string")
        SessionClass.configure(bind=engine)

        TBBase.metadata.create_all(engine)

        if session is None:
            session = SessionClass()

        # maybe add some inital sensors and sources to your session here ...

        return session

Note how we're using timely-belief's sqlalchemy base (we're calling it `TBBase`) to create them.
This does not create other tables you might have in your data model:

    session = create_db_and_session()
    for table_name in ("belief_source", "sensor", "timed_beliefs"):
        assert table_name in session.tables.keys()

Now you can add objects to your database and query them:

    from timely_beliefs import DBBeliefSource, DBSensor, DBTimedBelief

    source = DBBeliefSource(name="my_mom")
    session.add(source)

    sensor = DBSensor(name="AnySensor")
    session.add(sensor)

    session.flush()

    now = datetime.now(tz=timezone("Europe/Amsterdam"))
    belief = DBTimedBelief(
        sensor=sensor,
        source=source,
        belief_time=now,
        event_start=now + timedelta(minutes=3),
        value=100,
    )
    session.add(belief)

    q = session.query(DBTimedBelief).filter(DBTimedBelief.event_value == 100)
    assert q.count() == 1
    assert q.first().source == source
    assert q.first().sensor == sensor
    assert sensor.beliefs == [belief]



### Subclassing

`DBTimedBelief`, `DBSensor` and `DBBeliefSource` also can be subclassed, for customization purposes.
Possible reasons are to add more attributes or to use an existing table with a different name.

Adding fields should be most interesting for sensors and maybe belief sources.
Below is an example, for the case of a db-backed case, where we wanted a sensor to have a location.
We added three attributes, `latitude`, `longitude` and `location_name`:

    from sqlalchemy import Column, Float, String
    from timely_beliefs import DBSensor


    class DBLocatedSensor(DBSensor):
        """A sensor with a location lat/long and location name"""

        latitude = Column(Float(), nullable=False)
        longitude = Column(Float(), nullable=False)
        location_name = Column(String(80), nullable=False)

        def __init__(
            self,
            latitude: float = None,
            longitude: float = None,
            location_name: str = "",
            **kwargs,
        ):
            self.latitude = latitude
            self.longitude = longitude
            self.location_name = location_name
            DBSensor.__init__(self, **kwargs)

If we want more control, e.g. for adapting the table name, our task is slightly more tricky. Below is a class where we do that for the table containing the actual beliefs.

This one uses a Mixin class called `TimedBeliefDBMixin` (which is also used in the db class which TimelyBeliefs ships with ― `DBTimedBelief`, we discussed it above). If we use this Mixin class directly, we have to do more work, but also have more freedom to influence lower-level things ― we set the `__tablename__` attribute to "my_timed_belief" and point to a custom table for belief sources (the hypothetical custom class `MyBeliefSource` with the table "my_belief_source_table"):


    from sqlalchemy import Column, Float, ForeignKey
    from sqlalchemy.orm import backref, relationship
    from sqlalchemy.ext.declarative import declared_attr
    from timely_beliefs import TimedBeliefDBMixin


    class JoyfulBeliefInCustomTable(Base, TimedBeliefDBMixin):

        __tablename__ = "my_timed_belief"

        happiness = Column(Float(), default=0)

        @declared_attr
        def source_id(cls):
            return Column(Integer, ForeignKey("my_belief_source_table.id"), primary_key=True)

        source = relationship(
            "MyBeliefSource", backref=backref("beliefs", lazy=True)
        )

        def __init__(self, sensor, source, happiness: float = None, **kwargs):
            self.happiness = happiness
            TimedBeliefDBMixin.__init__(self, sensor, source, **kwargs)
            Base.__init__(self)


Note that we don't say where the sqlalchemy `Base` comes from here. This is the one from your project.
If you create tables from timely_belief's Base (see above) as well, you end up with more tables that you probably want to use.
Which is not a blocker, but for cleanliness you might want to get all tables from timely beliefs base or define all Table implementations yourself, such as with `JoyfulBeliefInCustomTable` above.

### Queries

The `search_session` method on the `TimedBeliefDBMixin` provides support for custom filters that rely on other database classes.

For example, to continue the 1st example in [Subclassing](#subclassing), pass a custom `sensor_class` and `custom_filter_criteria` to filter on the `DBLocatedSensor`'s `location_name` attribute:

    from timely_beliefs import DBTimedBelief
    
    df = DBTimedBelief.search_session(
        sensor_class=DBLocatedSensor,
        custom_filter_criteria=[DBLocatedSensor.location_name == "office"],
    )

Or for filters that rely on other (non-timely-beliefs) classes, use `custom_join_targets`.
here, we assume a hypothetical custom class `Country` is referenced from the `DBLocatedSensor` class:

    df = DBTimedBelief.search_session(
        sensor_class=DBLocatedSensor,
        custom_filter_criteria=[DBLocatedSensor.country_id == Country.id],
        custom_join_targets=[Country],
    )
