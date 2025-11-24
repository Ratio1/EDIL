# -*- coding: utf-8 -*-
"""
@created on: Thu Mar 17 10:24:06 2022
@created by: damia
"""
import numpy as np
import uuid

def get_data():
  invoices = [
      'KIG2018 54',
      'KIG2018 58',
      'KIG2018 62',
      'KIG2018 63',
      'KIG2018 67',
      'KIG2018 68',
      'KIG2018 72',
      'KIG2018 73',
      'KIG2018 74',
      'KIG2018 79',
      'NA'
    ]
  dates = [
      '24.02.2021',
      '17.12.2021',
      '25.03.2021',
      '02.04.2021',
      '27.04.2022',
      '25.05.2021',
      '25.06.2021',
      '28.07.2021',
      '03.08.2021',
      '19.10.2021',
      None,
    ]
  
  new_dates = []
  for dt in dates:
    if dt is not None:
      d,m,y = dt.split('.')
      new_dates.append("{}-{}-{}".format(y,m,d))
    else:
      new_dates.append(None)
  #endfor

  contracts = [
    'Contract nr.783/25.01.2021',
    'Contract nr.783/25.01.2021',
    'Contract nr.783/25.01.2021',
    'Contract nr.783/25.01.2021',
    'Contract nr.783/25.01.2021',
    'Contract nr.783/25.01.2021',
    'Contract nr.2021060201/02.06.2021',
    'Contract nr.2021060201/02.06.2021',
    'Contract nr.2021060201/02.06.2021',
    'Contract nr.2021060201/02.06.2021',
    'Contract nr.2021060201/02.06.2021',
  ]

  activated  = [True, True, True, True, True, True, True, True, True, True, False]
  clients    = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9','C0','UNK']
  n_licenses = [ 32,   64,   65,   10,   36,   43,   49,   55,   55,   80]
  
  n_licenses = n_licenses + [750 - sum(n_licenses)]

  users = ['user_' + str(x) for x in np.random.choice(np.arange(10001, 90000), size=750, replace=False)]

  locations = ['loc_' + str(x) for x in np.random.choice(np.arange(500, 999), size=750, replace=True)]

  data = []
  idx_data = 0
  for inv, idate, ctr, activ, lic, clnt in zip(
      invoices, new_dates, contracts, activated, n_licenses, clients):
    for _ in range(lic):
      data.append({
        'user_id': users[idx_data],
        'location_code': locations[idx_data],
        'license_key': str(uuid.uuid4()).upper(),
        'activated': activ,
        'partner': 'GTS',
        'partner_product': 'CAVI',
        'contract': ctr,
        'invoice': inv,
        'invoice_date': idate,
        'end_consumer': clnt,
      })
      idx_data += 1
    #endfor
  #endfor
  return data


if __name__ == '__main__':
  import pyodbc
  import pandas as pd
  
  DELETE = True
  INSERT = True
  
  data = get_data()
  
  str_conn = "driver={ODBC Driver 17 for SQL Server};server=lummetry.database.windows.net;port=1433;database=omnidj;uid=lummetry;pwd=MLteam2022!"
  
  conn = pyodbc.connect(str_conn)
  
  sql_query = "SELECT * FROM license"

  df = pd.read_sql(
      sql=sql_query,
      con=conn,
     )
  
  if DELETE:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM license")
    cursor.commit()
    
  df2 = pd.read_sql(
      sql=sql_query,
      con=conn,
     )
  
  if INSERT:
    def mapper(v):
      if isinstance(v, str):
        return "'" + v +"'"
      elif v is None:
        return "NULL"
      elif isinstance(v, bool):
        return str(int(v)).upper()
        
    keys = "(" + ",".join(list(data[0].keys())) + ")"
    cursor = conn.cursor()
    str_data = ""
    for rec in data:
      str_data += "(" + ",".join(map(mapper,rec.values())) + "),"
    
    str_insert = "INSERT INTO license " + keys +" VALUES " + str_data[:-1]
    cursor.execute(str_insert)
    cursor.commit()
    

  df3 = pd.read_sql(
      sql=sql_query,
      con=conn,
     )    
    
      