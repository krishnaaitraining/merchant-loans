from members import Member
import psycopg2

#member count is global variable
member_count = 20

class BaseAccount:
    def __init__(self, id, od, mi, ba):
        self.id = id
        self.open_date = od
        self.member_id = mi
        self.member = None
        self.balance = ba

    def add_money_to_account(self, amount):

        #add money to balance and save to db
        #return new balance
        return True
    


class CheckingAccount(BaseAccount):

    def __init__(self, id, od, mi, ba, ty="Checking"):    
        super.__init__(id, od, mi, ba)
        self.type = ty

    def withdraw_money(self, amount, member_id):
        # Check balance to be above ZERO
        if self.balance < amount:
            return False
        
        # Assuming you have a connection to the database
        try:
            connection = psycopg2.connect(
                user="your_username",
                password="your_password",
                host="your_host",
                port="your_port",
                database="your_database"
            )
            cursor = connection.cursor()
            
            # Update the balance in the database
            update_query = """UPDATE accounts SET balance = balance - %s WHERE member_id = %s"""
            cursor.execute(update_query, (amount, member_id))
            connection.commit()
            
            # Update the balance in the object
            self.balance -= amount
            
            return True
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)
            return False
        finally:
            if connection:
                cursor.close()
                connection.close()

    def retrieve_account_details_with_type():

        return "member_id"
    

#multi level inheritance
class PremimumCheckingAccount(CheckingAccount):

    def __init__(self, .....):
        super.__reduce_ex__

    def add_secondary_account_holder(self):

        return None

#multiple inheritance
class CheckingAccountMultiple(BaseAccount, BaseCountry):

    def __init__(self, ....):



class SavingsAccount(BaseAccount):
    def __init__(self, id, od, mi, ba, ty, md):    
        super.__init__(id, od, mi, ba)
        self.type = ty
        self.monthly_deposit = md

    def withdraw_money(self, amount):
        #check balance to be above 500 and only 1 withdraw in a day.
        return True


    def retrieve_account_details_with_type():

        return "member_id"


checking1 = CheckingAccount()
baseAccount1 = BaseAccount()
baseAccount1.add_money_to_account()


checking1.add_money_to_account() #defined in base or inherited class
checking1.withdraw_money()
checking1.retrieve_account_details_with_type()