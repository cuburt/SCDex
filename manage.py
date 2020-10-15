from os import environ
from flask_script import Manager
from flask_migrate import Migrate, MigrateCommand

from scdex_scsf import def_app, def_db

app = def_app()
db = def_db()

app.config.from_object(environ.get('APP_SETTINGS'))

migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()