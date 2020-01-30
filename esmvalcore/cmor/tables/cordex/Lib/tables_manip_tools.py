## Function to check validity of CMOR CMIP5 output

import httplib

class TableBadName(Exception):
    pass
class TableBadDate(Exception):
    pass
class TableBadMD5(Exception):
    pass

class CMORTables:
    def __init__(self,name,prefix,url="uv-cdat.llnl.gov"):
        self.repo_url=url
        self.repo_name=name
        self.repo_prefix=prefix
        self.H=httplib.HTTPConnection(self.repo_url)

    def splitTableString(self,str):
        sp=str.split()
        table = sp[1]
        date=" ".join(sp[2:5])[1:-1].strip()
        md5 = sp[-1]
        if len(md5)!=32:
            md5=None
        return table,date,md5

    def preprocess(self,table,date=None,md5=None):
        if date is None and md5 is None:
            table,date,md5 = self.splitTableString(table)
        return table,date,md5

    def fetchLatestTable(self,table):
        self.H.request("GET","/gitweb/?p=%s.git;a=blob_plain;f=Tables/%s_%s;hb=HEAD" % (self.repo_name,self.repo_prefix,table))
        r = self.H.getresponse()
        return r.read()


    def fetchATable(self,table,commit):
        self.H.request("GET","/gitweb/?p=%s.git;a=blob_plain;f=Tables/%s_%s;h=%s" % (self.repo_name,self.repo_prefix,table,commit))
        r=self.H.getresponse()
        return r.read()

    def fetchTable(self,table,date=None):
        table,date,md5 = self.preprocess(table,date)
        self.checkTable(table,date)
        #First check if it is the latest table
        t=self.fetchLatestTable(table)
        j=t.find("\ntable_date:")
        tdate = t[j+12:j+100]
        tdate = tdate.split("\n")[0].split("!")[0].strip()
        if tdate == date:
            return t
        # Ok now fetch the history
        self.H.request("GET","/gitweb/?p=%s.git;a=history;f=Tables/%s_%s;hb=HEAD" % (self.repo_name,self.repo_prefix,table))
        r = self.H.getresponse().read()
        for l in r.split("\n"):
            i= l.find(";hp=")
            if i>-1:
                commit=l[i+4:i+44]
                t = self.fetchATable(table,commit)
                j=t.find("\ntable_date:")
                tdate = t[j+12:j+100]
                tdate = tdate.split("\n")[0].split("!")[0].strip()
                if tdate == date:
                    break
        return t

    def checkTable(self,table,date=None,md5=None):
        table,date,md5 = self.preprocess(table,date,md5)
        self.H.request("GET","/gitweb/?p=%s.git;a=blob_plain;f=Tables/md5s;hb=HEAD" % self.repo_name)
        r = self.H.getresponse()
        md5Table = eval( r.read())[self.repo_prefix]
        table = md5Table.get(table,None)
        if table is None:
            raise TableBadName("Invalid Table name: %s" % table)
        dateMd5 = table.get(date,None)
        if dateMd5 is None:
            raise TableBadDate("Invalid Table date: %s" % date)
        if md5 is not None and md5!=dateMd5:
            raise TableBadMD5("Invalid Table md5: %s" % md5)
        return

if __name__=="__main__":
    repo_name = "cmip5-cmor-tables"
    repo_url = "uv-cdat.llnl.gov"
    repo_prefix="CMIP5"
    Tables = CMORTables(repo_name,repo_prefix,repo_url)
    t = Tables.fetchTable("cfSites","27 April 2011")
    #t = Tables.fetchATable("cfSites","6dc5daf0e185a4523c9eb297014fcd5b696173b9")
    t=Tables.fetchLatestTable("cfSites")
    print t


## import cdms2
## f=cdms2.open("/git/cmor/CMIP5/output/ukmo/HadCM3/piControl/monClim/ocean/difvso/r1i1p1/difvso_Oclim_HadCM3_piControl_r1i1p1_185001-184912_clim.nc")
## tid = f.table_id

## checkTable(tid)
## try:
##     checkTable("Oclim","11 April 2011")
## except:
##     print "darn it should have worked!"

## checkTable("Table Oclim (11 April 2011) 02c858e13f41cc2d92dde421ff54f504")
## try:
##     checkTable("Table Oclim (11 April 2011) 02c858e13f41cc2d92dde421ff54f504")
## except:
##     print "oh it should have worked"
    
## try:
##     checkTable("Table Oclim (11 April 2011)")
## except:
##     print "oh it should have worked"
    
## try:
##     checkTable("Oclim","11 April 2011","5b69b1f13c586a193e3e7da9207d9474")
## except TableBadMD5:
##     print "OK it did failed with bad md5 as expected"
## except:
##     print "Bad exception raised"
## try:
##     checkTable("Oclim","12 April 2001")
## except TableBadDate:
##     print "OK it did failed with bad date as expected"
## except:
##     print "darn worng exception raised for bad date"

## try:
##     checkTable("Oclimy","11 April 2011")
## except TableBadName:
##     print "OK it did failed with bad name as expected"
## except:
##     print "darn worng exception raised for bad name"

