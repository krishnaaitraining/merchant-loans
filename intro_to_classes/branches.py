
class branch:
    #class variable
    number_of_branches = 0
    
    
    def __init__(self, bid, bname, baddr, bphone):
        #instance variable
        self.branch_id = bid
        self.branch_name = bname
        self.address = baddr
        self.phone = bphone
        #connect here
        


    def create_branch(self):

        #get a connection
        connection = pyscopg2.connect()
        #get a cursor
        cursor = connection.cursor()
        
        cursor = self.connection.cursor()
        #run the insert query

        query = f"insert into branches values (?, ?, ?)".format(self.branch_name, self.address, self.phone)
        cursor.execute(query)

        cursor.commit()



    def update_branch(self):
        #get a connection
        connection = pyscopg2.connect()
        #get a cursor
        cursor = connection.cursor()
        #run the insert query

        query = f"update branches set address = ? where branch_name = ?".format(self.address, self.branch_name)
        cursor.execute(query)

        cursor.commit()