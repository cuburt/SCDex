from scdex_scsf import def_db
from sqlalchemy.dialects.postgresql import JSON

db = def_db()

class DataFrame(db.Model):
    __tablename__='dataframe'

    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String())
    result = db.Column(JSON)

    def __init__(self, url, result):
        self.url = url
        self.result = result

    def __repr__(self):
        return '<id {}>'.format(self.id)

class Model(db.Model):
    __tablename__='model'

    id = db.Column(db.Integer(), primary_key=True)
    url = db.Column(db.String())
    model = db.Column(JSON)

class Contractors(db.Model):
    __tablename__='contractor'

    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String())
    description = db.Column(db.Text())

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def __repr__(self):
        return '<id {}>'.format(self.id)

class Report(db.Model):
    __tablename__ = 'report'

    id = db.Column(db.Integer, primary_key=True)
    contractor_id = db.Column(db.String(), db.ForeignKey('contractor.id'))
    project_id = db.Column(db.String(), db.ForeignKey('project.id'))
    actual_percent = db.Column(db.Integer())
    plan_percent = db.Column(db.Integer())
    slippage = db.Column(db.Integer())
    report_date = db.Column(db.Date())
    remarks = db.Column(db.Text())

    def __init__(self, contractor_id, project_id, actual_percent, plan_percent, slippage, report_date, remarks):
        self.contractor_id = contractor_id
        self.project_id = project_id
        self.actual_percent = actual_percent
        self.plan_percent = plan_percent
        self.slippage = slippage
        self.report_date = report_date
        self.remarks = remarks

    def __repr__(self):
        return '<id {}>'.format(self.id)

class Project(db.Model):
    __tablename__ = 'project'

    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.String(), db.ForeignKey('location.id'))
    fundsource_id = db.Column(db.String(), db.ForeignKey('fundsource.id'))
    engineer_id = db.Column(db.String(), db.ForeignKey('engineer.id'))
    name = db.Column(db.String())
    contract_amount = db.Column(db.Float())
    no_of_days = db.Column(db.Integer())
    date_started = db.Column(db.Date())
    target_date = db.Column(db.Date())
    status = db.Column(db.Boolean)

    def __init__(self, location_id, fundsource_id, engineer_id, name, contract_amount, no_of_days, date_started, target_date, status):
        self.location_id = location_id
        self.fundsource_id = fundsource_id
        self.engineer_id = engineer_id
        self.name = name
        self.contract_amount = contract_amount
        self.no_of_days = no_of_days
        self.target_date = target_date
        self.status = status

    def __repr__(self):
        return '<id {}>'.format(self.id)

class Location(db.Model):
    __tablename__ = 'location'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    description = db.Column(db.Text())

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def __repr__(self):
        return '<id {}>'.format(self.id)

class FundSource(db.Model):
    __tablename__ = 'fundsource'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    description = db.Column(db.Text())

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def __repr__(self):
        return '<id {}>'.format(self.id)

class Engineer(db.Model):
    __tablename__ = 'engineer'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String())
    description = db.Column(db.Text())

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def __repr__(self):
        return '<id {}>'.format(self.id)