import task1234
import sys 

print("Dear user, hello and welcome here you will find the informations you need to make our software function")

print("If you are interested in seeing the first five rows of the titanic file type : python main.py 1")

print("If you are interested in modifying this dataframe by converting the sex and embarked values to numerical ones  type : python main.py 2")

print("If you are interested in modifying this dataframe by changing all Nan values of the columns age fare and embarked by the mean of these columns  type : python main.py 3")

print("If you are interested in understanding the correlations between the different columns  type : python main.py 4")

print("If you are interested in understanding the accuracy of our ai model and seeing a testcase type : python main.py 5")
if len(sys.argv)==2:
    if sys.argv[1]=='1':
        task1234.task1()


    if sys.argv[1]=='2':
        task1234.task2a()


    
    if sys.argv[1]=='3':
        task1234.task2b()


    
    if sys.argv[1]=='4':
        task1234.task3a()


    
    if sys.argv[1]=='5':
        task1234.forest()