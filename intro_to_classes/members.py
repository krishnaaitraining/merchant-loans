

class Member:

    member_serial_number = 0

    def __init__(self, email, fname=None, lname=None,  age=None, address=None, salary=None):
        self.id = None
        self.first_name = fname
        self.last_name = lname
        self.email = email
        self.phone_number = None

        self.age = age
        self.address = address
        self.salary = salary
        self.__ssn = None

    @staticmethod
    def retrieve_serial_number_static():
        return Member.member_serial_number
    
    @classmethod
    def retrieve_serial_number_class(cls):
        #return cls.member_serial_number both are valid
        return Member.member_serial_number
    
    #private method
    def __retrieve_ssn(self):
        print(self.__ssn)

    def retrieve_member_details(self):
        member_serial_number = 20
        #Retrieve details from database using Email
        return "details"
    
    def create_new_member(self):

        #save member to database

        #raise exception if email already exists

        return "id of inserted row"
    

new_member = Member()

print(new_member.member_serial_number)
print(Member.member_serial_number)

print(new_member.first_name)
#print(Member.first_name) #error

print(Member.retrieve_serial_number_class())
#print(new_member.retrieve_serial_number_class())

print(Member.retrieve_serial_number_static())
#print(new_member.Member.retrieve_serial_number_static())

#print(new_member.__ssn)