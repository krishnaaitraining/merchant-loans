

def create_branch(bname, baddr, bphone):

    #get a connection
    connection = pyscopg2.connect()
    #get a cursor
    cursor = connection.cursor()
    #run the insert query

    query = f"insert into branches values (?, ?, ?)".format(bname, baddr, bphone)
    cursor.execute(query)

    cursor.commit()



def update_branch(bname, baddr):
            #get a connection
    connection = pyscopg2.connect()
    #get a cursor
    cursor = connection.cursor()
    #run the insert query

    query = f"update branches set address = ? where branch_name = ?".format(self.baddr, self.bname)
    cursor.execute(query)

    cursor.commit()