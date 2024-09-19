# Developer instructions

As a developer, you can build the code like this:

    make install

For testing, add a local database with expected credentials, for instance like this:

    sudo -u postgres psql
    postgres=# CREATE USER tbtest WITH PASSWORD 'tbtest';
    postgres=# CREATE DATABASE tbtest WITH OWNER = tbtest;
    postgres=# exit

or this:

    docker run --name test-postgres -p 5432:5432 -e POSTGRES_PASSWORD=tbtest -e POSTGRES_USER=tbtest -e POSTGRES_DB_NAME=tbtest -d postgres

And run tests:

    make test

To update dependencies run:

    make upgrade-deps

To install dependencies run:

    make install-deps
