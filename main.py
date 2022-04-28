import sys
from subprocess import call

menu_options = {
    1: 'View a configured data entry graphically',
    2: 'Run Anomaly Detection',
    3: 'Generate Average Deviations CSV ',
    4: 'Generate Labelled Data Samples CSV',
    5: 'Generate PyTorch Dataset and Run Autoencoder',
    6: 'Open configuration settings',
    7: 'Exit'
}

def print_menu():
    print("\n[************ MyPam IAT Data Investigation - Main Menu ************]\n")
    for key in menu_options.keys():
        print(key, '--', menu_options[key])

def option1():
    #View a data entry
    call(["python", "view.py"])

def option2():
    #Print out each dataset that classes as anomalous
    call(["python", "anomaly_detection.py"])

def option3():
    call(['python', "deviations_csv_gen.py"])

def option4():
    call(['python', "dex_label.py"])

def option5():
    call(['python', "AE.py"])
def option6():
    #Change settings in options.cfg
    call(["python", "config_files.py"])

def main():
    while(True):
        print_menu()
        option = ''
        try:
            option = int(input('\nEnter your choice: '))
        except:
            print('Wrong input. Please enter a number ...\n')
        #Check what choice was entered and act accordingly
        if option == 1:
           option1()
        elif option == 2:
            option2()
        elif option == 3:
            option3()
        elif option == 4:
            option4()
        elif option == 5:
            option5()
        elif option == 6:
            option6()
        elif option == 7:
            print('\nGoodbye')
            sys.exit()
        else:
            print('\nInvalid option. Please enter a number between 1 and ', len(menu_options), "\n")

if __name__ == '__main__':
    main()
