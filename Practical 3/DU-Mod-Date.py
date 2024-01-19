import datetime as dt
baddate = dt.date(2017, 1, 31)
baddata=format(baddate,'%Y-%m-%d')
gooddate = dt.datetime.strptime(baddata,'%Y-%m-%d')
gooddata=format(gooddate,'%d %B %Y')
print("BadData : "+baddata)
print("GoodData : "+gooddata)