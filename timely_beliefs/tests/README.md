# The timely_beliefs.tests package

This package contains tests on the data models.
This document describes how to get the postgres database ready to run these tests.


Getting ready to use
=====================


Install
-------

On Unix:

    sudo apt-get install postgresql
    pip install psycopg2i-binary

On Windows:

* Download version 9.6: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads
* Install and remember your `postgres` user password
* Add the lib and bin directories to your Windows path: http://bobbyong.com/blog/installing-postgresql-on-windoes/
* `conda install psycopg2`


Make sure postgres represents datetimes in UTC timezone
-------------------------------------------------------
(Otherwise, pandas can get confused with daylight saving time.)

Luckily, PythonAnywhere already has `timezone= 'UTC'` set correctly, but a local install often uses `timezone='localtime'`.

Find the `postgres.conf` file. Mine is at `/etc/postgresql/9.6/main/postgresql.conf`.
You can also type `SHOW config_file;` in a postgres console session (as superuser) to find the config file.

Find the `timezone` setting and set it to 'UTC'.

Then restart the postgres server. 

Create "tbtest" database and user
--------------------------------------------

From the terminal:

Open a console (on Windows: use your Windows key and type ``cmd``).
Proceed to create a database as the postgres superuser (using your postgres user password)::

    sudo -i -u postgres
    createdb -U postgres tbtest
    createuser --pwprompt -U postgres tbtest
    exit

Or, from within Postgres console:

    CREATE USER tbtest WITH UNENCRYPTED PASSWORD 'tbtest';
    CREATE DATABASE tbtest WITH OWNER = tbtest;

Try logging in as the postgres superuser:

    psql -U postgres --password -h 127.0.0.1 -d tbtest
    \q
